#include <iostream>

#include "dune/grid/config.h"
#include "dune/grid/uggrid.hh"
#include <dune/istl/operators.hh>

#include "fem/assemble.hh"
#include "fem/spaces.hh"
#include "fem/diffops/elasto.hh"
#include "fem/embedded_errorest.hh"
#include "fem/iterate_grid.hh"
#include "io/vtk.hh"
#include "linalg/direct.hh"
#include "linalg/apcg.hh"                   // Pcg preconditioned CG method
#include "mg/additiveMultigrid.hh"          // makeBPX MG preconditioner
#include "linalg/threadedMatrix.hh"
#include "linalg/triplet.hh"
#include "utilities/enums.hh"
#include "utilities/gridGeneration.hh"
#include "utilities/kaskopt.hh"
#include "utilities/memory.hh"             // for mmoveUnique 
#include "utilities/timing.hh"

using namespace Kaskade;
#include "adaBeam.hpp"


int main(int argc, char *argv[])
{
  using namespace boost::fusion;


  Timings& timer=Timings::instance();
  timer.start("total computing time");
  std::cout << "Start elastic beam tutorial program." << std::endl;
  
  int maxit, order, refinements, solver, verbose, materialList;
  bool vtk, linearStrain;
  double atol, L, W, aTolx;
  std::string material, storageScheme("A");
  
  if (getKaskadeOptions(argc,argv,Options
    ("L",                L,                4.,          "length of the elastic beam")
    ("W",                W,                0.2,        "width/ of the elastic beam")
    ("refinements",      refinements,      0,          "number of uniform grid refinements")
    ("order",            order,            1,          "finite element ansatz order")
    ("material",         material,         "steel",    "type of material")
    ("linearStrain",     linearStrain,     true,       "true= linear strain tensor, false=nonlinear strain tensor")
    ("aTolx",            aTolx,            1e-4,       "absolute error tolerance for embedded error estimator")
    ("solver",           solver,           0,          "0=UMFPACK, 1=PARDISO 2=MUMPS 3=SUPERLU 4=UMFPACK32/64 5=UMFPACK64")
    ("verbose",        verbose,          0,          "amount of reported details")
    ("vtk",              vtk,              false,       "write solution to VTK file")
    ("atol",             atol,             1e-8,       "absolute energy error tolerance for iterative solver")
    ("maxit",            maxit,            100,        "maximum number of iterations")
    ("MaterialList",     materialList,      0,         "Report the known materials in kaskade")
    ))
    return 1;

  if(materialList>0){
    auto materials = Elastomechanics::ElasticModulus::materials();
    printMaterial(materials);
  }

  std::cout << "refinements of original mesh : " << refinements << std::endl;
  std::cout << "discretization order         : " << order << std::endl;
////  domain
  constexpr int DIM = 3;
  using Grid = Dune::UGGrid<DIM>;

  timer.start("computing time for grid creation & refinement");

  // set dimensions of the bar
  Dune::FieldVector<double,DIM> x0(0.0), dx(W); dx[0] = L; 
  Dune::FieldVector<double,DIM> dh(0.2); dh[0] = 0.6;

////  grid
  GridManager<Grid> gridManager( createCuboid<Grid>(x0, dx, dh, true) );
  // mesh refinement
  gridManager.globalRefine(refinements);
  gridManager.enforceConcurrentReads(true);

  timer.stop("computing time for grid creation & refinement");

///// spaces 
  timer.start("computing time for functional setup");

  using Spaces = boost::fusion::vector<H1Space<Grid> const*>;
  
  // construction of finite element space for the vector solution u.
  H1Space<Grid> h1Space(gridManager,gridManager.grid().leafGridView(),order);

  Spaces spaces(&h1Space);

//// variables
  using VariableDescriptions = boost::fusion::vector<VariableDescription<0,3,0> >;
  using VariableSet = VariableSetDescription<Spaces,VariableDescriptions>;

  std::string varNames[1] = { "u" };
  VariableSet variableSet(spaces, varNames);

  using CoefficientVectors = decltype(variableSet.zeroCoefficientVector());

//// functional 
  // use linear or nonlinear strain tensor
  using StrainTensor = LinearizedGreenLagrangeTensor<double,DIM>;    
  using Functional = ElasticityFunctional<VariableSet,StrainTensor>;
  using Assembler = VariationalFunctionalAssembler<LinearizationAt<Functional> >;
  // using CoefficientVectors = VarSetDesc::CoefficientVectorRepresentation<0,1>::type;

  // Create the variational functional.
  Functional F(ElasticModulus::material(material));

  timer.stop("computing time for functional setup");

  constexpr int neq = Functional::TestVars::noOfVariables;
  constexpr int nvars = Functional::AnsatzVars::noOfVariables;
  std::cout << "no of variables = " << nvars << std::endl;
  std::cout << "no of equations = " << neq   << std::endl;
  
  Assembler assembler(spaces);
  
  size_t nnz  = assembler.nnz(0,neq,0,nvars);
  size_t size = variableSet.degreesOfFreedom(0,nvars);
  if ( verbose>0) std::cout << "init mesh: nnz = " << nnz << ", dof = " << size << std::endl;

  std::vector<std::pair<double,double> > tolX(nvars);
  double rTolx = 0;
  for (int i=0; i<tolX.size(); ++i) {
    tolX[i] = std::make_pair(aTolx,rTolx);
  }
  std::cout << std::endl << "Accuracy: atol = " << aTolx << ",  rtol = " << rTolx << std::endl;

  bool accurate = false;
  int refSteps = -1;
  int iter=0;
  // just to use beyond loop
  VariableSet::VariableSet xx(variableSet);
  do {
    refSteps++;

    // construct Galerkin representation
    timer.start("computing time for assemble " + std::to_string(iter));

    VariableSet::VariableSet x(variableSet);
    
    assembler.assemble(linearization(F,x));

    CoefficientVectors solution = variableSet.zeroCoefficientVector();
    CoefficientVectors rhs(assembler.rhs());

    timer.stop("computing time for assemble " + std::to_string(iter));

    timer.start("computing time for solve " + std::to_string(iter));
    using X = Dune::BlockVector<Dune::FieldVector<double,DIM>>;
    DefaultDualPairing<X,X> dp;
    using Matrix = NumaBCRSMatrix<Dune::FieldMatrix<double,DIM,DIM>>;
    using LinOp = Dune::MatrixAdapter<Matrix,X,X>;
    Matrix Amat(assembler.get<0,0>(),true);
    LinOp A(Amat);
    SymmetricLinearOperatorWrapper<X,X> sa(A,dp);
    PCGEnergyErrorTerminationCriterion<double> term(atol,maxit);
    
    Dune::InverseOperatorResult res;
    X xi(component<0>(rhs).N());

    std::unique_ptr<SymmetricPreconditioner<X,X>> mg;

    if (order==1)
    {
      throw std::invalid_argument("order 1 not supported\n");
      return -1;
    } 
    
    
    H1Space<Grid> p1Space(gridManager,gridManager.grid().leafGridView(),1);
    if (storageScheme=="A")
      mg = moveUnique(makePBPX(Amat,h1Space,p1Space,DenseInverseStorageTag<double>(),gridManager.grid().maxLevel()));
    else if (storageScheme=="L")
      mg = moveUnique(makePBPX(Amat,h1Space,p1Space,DenseCholeskyStorageTag<double>(),gridManager.grid().maxLevel()));
    else
    {
      std::cerr << "unknown storage scheme provided\n";
      return -1;
    }

    Pcg<X,X> pcg(sa,*mg,term,verbose);
    pcg.apply(xi,component<0>(rhs),res);
    std::cout << "PCG iterations: " << res.iterations << "\n";
    xi *= -1;
    component<0>(x) = xi;
    timer.stop("computing time for solve " + std::to_string(iter));

    timer.start("computing time for refinement " + std::to_string(iter));
    VariableSet::VariableSet e = x;
    projectHierarchically(e);
    e -= x;    
  
    accurate = embeddedErrorEstimator(variableSet,e,x,IdentityScaling(),tolX,gridManager,verbose);
    nnz = assembler.nnz(0,1,0,1);;
    size_t size = variableSet.degreesOfFreedom(0,1);
    if (verbose>0) 
      std::cout << "new mesh: nnz = " << nnz << ", dof = " << size << std::endl;
    
    timer.stop("computing time for refinement " + std::to_string(iter));

    // VariableSet::VariableSet xx may be used beyond the do...while loop	
    xx.data = x.data;
    iter++; 
    if (iter>10) 
    {
      std::cout << "*** Maximum number of iterations exceeded ***" << std::endl;
      break;
    }
    
    
  } while (not accurate);
  


  timer.start("postprocessing time");
  //Postprocessing 
  L2Space<Grid> l2Space(gridManager,gridManager.grid().leafGridView(),0);

  L2Space<Grid>::Element_t<DIM> normalStress(l2Space);
  interpolateGlobally<PlainAverage>(normalStress,makeFunctionView(h1Space, [&] (auto const& evaluator)
  {
    auto modulus = ElasticModulus::material(material);
    HyperelasticVariationalFunctional<Elastomechanics::MaterialLaws::StVenantKirchhoff<DIM>,StrainTensor>
    energy(modulus);

    energy.setLinearizationPoint(component<0>(xx).derivative(evaluator));
    auto stress = Dune::asVector(energy.cauchyStress());

    return Dune::FieldVector<double,3>{stress[0],stress[4],stress[8]};
  }));

   auto vsd = makeVariableSetDescription(makeSpaceList(&l2Space, &h1Space),
                                        boost::fusion::make_vector(Variable<SpaceIndex<0>,Components<3>>("NormalStress"),
                                                                   Variable<SpaceIndex<1>,Components<3>>("Displacement")));
  auto data = vsd.variableSet();
  component<0>(data) = normalStress;
  component<1>(data) = component<0>(xx);

  // output of solution in VTK format for visualization,
  // the data are written as ascii stream into file elasto.vtu,
  // possible is also binary
  if (vtk)
  {
    ScopedTimingSection ts("computing time for file i/o",timer);
    std::string outStr("elasto_");
    if(linearStrain)
      outStr.append("linear_p=");
    else
      outStr.append("nonlinear_p=");
    writeVTK(data,outStr+paddedString(order,1),IoOptions().setOrder(order).setPrecision(16));
  }
  timer.stop("postprocessing time");
  timer.stop("total computing time");
  if (verbose>0)
    std::cout << timer;
    
  std::cout << "You successfully ran the elastic beam tutorial program." << std::endl;
  
  return 0;
}

