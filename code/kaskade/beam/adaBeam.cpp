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
#include "io/vtkreader.hh"
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

  std::string default_cuboid_path = "/data/numerik/people/pvillani/thesis/data/d2/kaskade/default_cuboid.vtu";

  
  int maxit, order, refinements, max_refinements, solver;
  bool verbose, save_mesh;
  double atol, L, W, E, nu, tolx;
  std::string storageScheme("A"), meshfile, datapath;
  
  if (getKaskadeOptions(argc,argv,Options
    ("L",                L,                4.,         "length of the elastic beam")
    ("W",                W,                0.2,        "width/ of the elastic beam")
    ("nu",               nu,               0.3,        "Poisson's ratio")
    ("E",                E,                2.1e11,     "Young's modulus")
    ("tolx",             tolx,             1e-1,       "relative error tolerance for embedded error estimator")
    ("refinements",      refinements,      0,          "number of uniform grid refinements")
    ("max_refinements",  max_refinements,  10,         "maximum number of steps for adaptive mesh refinement")
    ("order",            order,            2,          "finite element ansatz order" )
    ("solver",           solver,           0,          "0=UMFPACK, 1=PARDISO 2=MUMPS 3=SUPERLU 4=UMFPACK32/64 5=UMFPACK64")
    ("verbose",          verbose,          false,      "amount of reported details")
    ("atol",             atol,             1e-8,       "absolute energy error tolerance for iterative solver")
    ("maxit",            maxit,            100,        "maximum number of solver iterations")
    ("mesh",             meshfile,         default_cuboid_path,         "input mesh file")
    ("save_mesh",        save_mesh,        false,      "save final mesh")
    ("datapath",         datapath,         ".",        "path to save data")
    ))
    return 1;

  if (order==1)
  {
    throw std::invalid_argument("order 1 not supported\n");
    return -1;
  } 
    
  
  constexpr int DIM = 3;

  int n_meas = 4;

  Dune::FieldVector<double,DIM> sensors[n_meas] = {
    // Dune::FieldVector<double,DIM>({L/4, W/2, W}), // top left
    // Dune::FieldVector<double,DIM>({L/2, W/2, W}), // top middle
    // Dune::FieldVector<double,DIM>({3*L/4, W/2, W}), // top right
    // Dune::FieldVector<double,DIM>({3*L/8, W/2, 0.0}), // low left
    // Dune::FieldVector<double,DIM>({5*L/8, W/2, 0.0}) // low right
    Dune::FieldVector<double,DIM>({L/2, W/2, W}), // top
    Dune::FieldVector<double,DIM>({L/2, W/2, 0.0}), // low
    Dune::FieldVector<double,DIM>({L/2, W, W/2}), // left
    Dune::FieldVector<double,DIM>({L/2, 0.0, W/2}) // right
  };

  double meas[n_meas] = {0.0};
  double residuals[max_refinements+1] [n_meas] = {0.0};
  double tolerances[max_refinements+1] = {0.0};

  if (refinements > 0)
    if (verbose)   
      std::cout << "refinements of original mesh : " << refinements << std::endl;

  if (verbose)
    std::cout << "discretization order         : " << order << std::endl;
////  domain
  using Grid = Dune::UGGrid<DIM>;

  // set dimensions of the bar
  Dune::FieldVector<double,DIM> x0(0.0), dx(W); dx[0] = L; 
  Dune::FieldVector<double,DIM> dh(0.2); dh[0] = 0.6;

////  grid
  std::string mesh_path;
  if (meshfile.empty())
  {
    mesh_path = default_cuboid_path;
  }
  else
  {
    mesh_path = meshfile;
  };

  VTKReader vtk(mesh_path);
  GridManager<Grid> gridManager(vtk.createGrid<Grid>());

  // mesh refinement
  gridManager.globalRefine(refinements);
  
  
  gridManager.enforceConcurrentReads(true);

///// spaces 

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
  ElasticModulus modulus = ElasticModulus();
  modulus.setYoungPoisson(E,nu);
  Functional F(modulus);


  constexpr int neq = Functional::TestVars::noOfVariables;
  constexpr int nvars = Functional::AnsatzVars::noOfVariables;
  if (verbose)
  {
    std::cout << "no of variables = " << nvars << std::endl;
    std::cout << "no of equations = " << neq   << std::endl;
  }
  
  Assembler assembler(spaces);
  
  size_t nnz  = assembler.nnz(0,neq,0,nvars);
  size_t size = variableSet.degreesOfFreedom(0,nvars);
  if ( verbose) std::cout << "init mesh: nnz = " << nnz << ", dof = " << size << std::endl;

  std::vector<std::pair<double,double> > tolX(nvars);
  double aTolx = 0;
  for (int i=0; i<tolX.size(); ++i) {
    tolX[i] = std::make_pair(aTolx,tolx);
  }
  if (verbose) 
    std::cout << std::endl << "Accuracy (relative tolerance): " << tolx << std::endl;

  bool accurate = false;
  int iter=0;
  // just to use beyond loop
  VariableSet::VariableSet xx(variableSet);

  do {

    // construct Galerkin representation

    VariableSet::VariableSet x(variableSet);
    
    assembler.assemble(linearization(F,x));

    CoefficientVectors solution = variableSet.zeroCoefficientVector();
    CoefficientVectors rhs(assembler.rhs());

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
    
    H1Space<Grid> p1Space(gridManager,gridManager.grid().leafGridView(),1);
    
    mg = moveUnique(makePBPX(Amat,h1Space,p1Space,DenseInverseStorageTag<double>(),gridManager.grid().maxLevel()));
    
    Pcg<X,X> pcg(sa,*mg,term,verbose);
    pcg.apply(xi,component<0>(rhs),res);
    if (verbose) 
      std::cout << "PCG iterations: " << res.iterations << "\n";

    xi *= -1;
    component<0>(x) = xi;

    VariableSet::VariableSet e = x;
    projectHierarchically(e);
    e -= x;    
  
    // VariableSet::VariableSet xx may be used beyond the do...while loop	
    xx.data = x.data;

    for (int i=0; i<n_meas; i++)
    {
      if (i < 2)
        residuals[iter][i] = component<0>(e).value(GlobalPosition<Grid>(sensors[i]))[2]; 
      else 
        residuals[iter][i] = component<0>(e).value(GlobalPosition<Grid>(sensors[i]))[1];
    }

    ErrorestDetail::GroupedSummationCollector<EmbeddedErrorestDetail::CellByCell> sum(EmbeddedErrorestDetail::CellByCell(variableSet.indexSet));
    scaledTwoNormSquared(join(VariableDescriptions(),VariableDescriptions()),
                          join(e.data,x.data),variableSet.spaces,Kaskade::IdentityScaling(),sum);

    int const s = variableSet.noOfVariables;
    int const n = sum.sums.shape()[0];
    
    assert(sum.sums.shape()[1]==2*s);
    
    // Compute scaled L2 norms of solution and errors.
    std::vector<double> norm2(s), error2(s);
    for (int idx=0; idx<n; ++idx)
      for (int j=0; j<s; ++j) {
        error2[j] += sum.sums[idx][j];
        norm2[j]  += sum.sums[idx][j+s];
      }      
    
    tolerances[iter] = sqrt( error2[0]/norm2[0]);

    iter++; 
    
    if (iter>max_refinements) 
    {
      if (iter > 1)
        std::cout << "*** Maximum number of refinement steps exceeded ***" << std::endl;
      break;
    }

    accurate = embeddedErrorEstimator(variableSet,e,x,IdentityScaling(),tolX,gridManager,verbose);
    nnz = assembler.nnz(0,1,0,1);;
    size_t size = variableSet.degreesOfFreedom(0,1);
    if (verbose) 
      std::cout << "new mesh: nnz = " << nnz << ", dof = " << size << std::endl;


  } while (not accurate);
  
  
  for (int i=0; i<n_meas; i++)
  {
    if (i < 2)
      meas[i] = component<0>(xx).value(GlobalPosition<Grid>(sensors[i]))[2];
    else
      meas[i] = component<0>(xx).value(GlobalPosition<Grid>(sensors[i]))[1];
  }
  // save meas
  std::ofstream measfile;
  measfile.open (datapath+"/measurements.csv");
  for (int i=0; i<n_meas-1; i++)
  {
    measfile << meas[i] << ",";
  }
  measfile << meas[n_meas -1] << std::endl;
  measfile.close();

  std::ofstream tolfile;
  tolfile.open (datapath+"/tolerances.csv");
  for (int i=0; i<iter-1; i++)
  {
    tolfile << tolerances[i]<< ",";
  }
  tolfile << tolerances[iter-1] << std::endl;
  tolfile.close();

  // save residuals
  std::ofstream resfile;
  resfile.open (datapath+"/residuals.csv");
  for (int i=0; i<iter; i++)
  {
    for (int j=0; j<n_meas-1; j++)
    {
      resfile << residuals[i][j] << ",";
    }
    resfile << residuals[i][n_meas-1] << std::endl;
  }
  resfile.close();


  //Postprocessing 
  L2Space<Grid> l2Space(gridManager,gridManager.grid().leafGridView(),0);


  L2Space<Grid>::Element_t<DIM> normalStress(l2Space);
  interpolateGlobally<PlainAverage>(normalStress,makeFunctionView(h1Space, [&] (auto const& evaluator)
  {
    ElasticModulus modulus = ElasticModulus();
    modulus.setYoungPoisson(E,nu);
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

  if (verbose)
  {
    std::cout << "Measured displacements = ";
    for (int i=0; i<n_meas; i++)
    {
      std::cout << meas[i] << " ";
    }
    std::cout << std::endl;
  }
  if (save_mesh)
    writeVTK( component<0>(xx),datapath+"/mesh", IoOptions().setOrder(order).setPrecision(16),"u");
  
  
  return 0;
}

