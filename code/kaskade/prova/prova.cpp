#include <iostream>

#include "dune/grid/config.h"
#include "dune/grid/uggrid.hh"

#include "fem/assemble.hh"
#include "fem/istlinterface.hh"
#include "fem/functional_aux.hh"
#include "fem/gridmanager.hh"
#include "fem/lagrangespace.hh"
#include "fem/variables.hh"

#include "io/vtk.hh"
#include "linalg/direct.hh"
#include "utilities/gridGeneration.hh"

using namespace Kaskade;
#include "prova.hpp"

int main()
{
  std::cout << "Start Laplacian tutorial program..." << std::endl;

  constexpr int dim = 2;
  constexpr double side_length = 1;
  constexpr int refinements = 5;
  constexpr int order = 2;
  constexpr double penalty = 1e6;

  // define the square grid
  using Grid = Dune::UGGrid<dim>;
  GridManager<Grid> temperatureGM( createUnitSquare<Grid>(side_length) );
  temperatureGM.globalRefine(refinements);

  // construction of finite element space for the scalar solution u.
  using LeafView = Grid::LeafGridView;
  using H1Space = FEFunctionSpace<ContinuousLagrangeMapper<double,LeafView>>;
  H1Space temperatureSpace(temperatureGM, temperatureGM.grid().leafGridView(), order);

  using Spaces = boost::fusion::vector<H1Space const*>;
  Spaces temperatureSpaces(&temperatureSpace);

  using VariableDescriptions = boost::fusion::vector<Variable<SpaceIndex<0>,Components<1>>>;
  using VariableSetDesc = VariableSetDescription<Spaces,VariableDescriptions>;
  VariableSetDesc temperatureSpacesVarSetDesc(temperatureSpaces,{ "u" });
  auto u = temperatureSpacesVarSetDesc.variableSet();

  // visualize the grid with the variable u initialized as zero everywhere
  writeVTK(u,"temperature_initial",IoOptions().setOrder(order).setPrecision(7));
  std::cout << "Grid and initial variable defined" << std::endl;


    // define a functional class which contains information associated with the Poisson's equation
  using Functional = HeatFunctional<double,VariableSetDesc>;
  Functional F(penalty);

  // assemble matrices and vectors which contain the information function space discretization and the differential equation
  using Assembler = VariationalFunctionalAssembler<LinearizationAt<Functional> >;
  Assembler assembler(temperatureSpaces);
  assembler.assemble(linearization(F,u));

  using Operator = AssembledGalerkinOperator<Assembler>;
  Operator A(assembler);

  using CoefficientVectors = VariableSetDesc::CoefficientVectorRepresentation<0,1>::type;
  CoefficientVectors solution = temperatureSpacesVarSetDesc.zeroCoefficientVector();
  CoefficientVectors rhs(assembler.rhs());

  // solve the system of linear equation and obtain the coefficients which represent the solution  
  directInverseOperator(A).applyscaleadd(-1.0,rhs,solution);
  component<0>(u) += component<0>(solution);

  // visualize the solution
  writeVTK(u,"temperature_output",IoOptions().setOrder(order).setPrecision(7));

  std::cout << "End Laplacian tutorial program" << std::endl;

}
