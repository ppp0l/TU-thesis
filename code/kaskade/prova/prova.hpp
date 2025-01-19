#include <algorithm>
#include "fem/diffops/boundaryConditions.hh"
#include "fem/functional_aux.hh"
#include "fem/fixdune.hh"
#include "fem/variables.hh"
#include "utilities/linalg/scalarproducts.hh"

// Simple Poisson equation -\Delta u = f with homogeneous Dirichlet boundary conditions
// on \Omega (which will be the unit square in our case).
// The corresponding variational functional on H^1_0(\Omega) is J = \int_\Omega |\nabla u|^2 - f*u dx.
// Let us call the integrand F(x,u,\nabla u) = |\nabla u|^2 - f*u.
template <class RType, class VarSet>
class HeatFunctional : public Kaskade::FunctionalBase<Kaskade::VariationalFunctional>
{
public:
  using Scalar = RType;
  using OriginVars = VarSet;
  using AnsatzVars = VarSet;
  using TestVars = VarSet;

  static constexpr int dim = AnsatzVars::Grid::dimension;
  static constexpr int uIdx = 0;
  static constexpr int uSpaceIdx = spaceIndex<AnsatzVars,uIdx>;

  // The domain cache defines F as well as its first and second directional derivative,
  // evaluated at the current iterate u.
  class DomainCache : public Kaskade::CacheBase<HeatFunctional,DomainCache>
  {
  public:
    DomainCache(HeatFunctional const&,
                typename AnsatzVars::VariableSet const& vars_,
                int flags=7):
      vars(vars_)
    {}


    template <class Position, class Evaluators>
    void evaluateAt(Position const& x, Evaluators const& evaluators)
    {
      u  = vars.template value<uIdx>(evaluators);
      du = vars.template derivative<uIdx>(evaluators);
      f = 1.0;
    }

    Scalar
    d0() const
    {
      return sp(du,du)/2 - f*u;
    }

    template<int row>
    D1Result<TestVars,row> d1(Kaskade::VariationalArg<Scalar,dim,TestVars::template Components<row>::m> const& arg) const
    {
      return sp(du,arg.derivative) - f*arg.value;
    }

    template<int row, int col>
    Scalar d2_impl(Kaskade::VariationalArg<Scalar,dim,TestVars::template Components<row>::m> const &arg1,
                   Kaskade::VariationalArg<Scalar,dim,AnsatzVars::template Components<row>::m> const &arg2) const
    {
      return sp(arg1.derivative,arg2.derivative);
    }

  private:
    typename AnsatzVars::VariableSet const& vars;
    Dune::FieldVector<Scalar,1> u, f;
    Dune::FieldVector<Scalar,dim> du;
    Kaskade::LinAlg::EuclideanScalarProduct sp;
  };

  // The boundary cache implements the boundary conditions. We do not implement Dirichlet boundary conditions
  // by eliminating degrees of freedom or equivalently modify the stiffness matrix and right hand side, but
  // by penalizing the deviation of u from the Dirichlet value ud (zero in our simple example), in the form of
  // \int_{\partial \Omega} g(x,u,\nabla u) ds.
  //
  // The boundary cache implements g and its first and second directional derivatives. Different penalizations
  // exist. The simplest is the quadratic Dirichlet penalty g = \gamma/2 * (u-ud)^2. For convergence on mesh
  // refinement, i.e. h->0, the penalty factor gamma must be scaled with the face diameter. This is provided
  // for convenience by the class DirichletPenaltyBoundary. A more sophisticated penalty formulation is
  // Nitsche's method, provided by the class DirichletNitscheBoundary. Here we use Nitsche's method on the
  // right edge of the unit square and the Dirichlet penalty method on the other sides.
  class BoundaryCache
  {
  public:
    BoundaryCache(HeatFunctional<RType,AnsatzVars> const& functional,
                  typename AnsatzVars::VariableSet const& vars_,
                  int flags=7):
      vars(vars_), penalty(functional.penalty), uDirichletBoundaryValue(0.0)
    {}

    template <class FaceIterator>
    void moveTo(FaceIterator const& fi)
    {
      useNitsche = fi->centerUnitOuterNormal()[0] > 0.5;
      if (useNitsche)
        nitsche.moveTo(fi);
      else
        dirichlet.moveTo(fi);
    }

    template <class Position, class Evaluators>
    void evaluateAt(Position const& xi, Evaluators const& evaluators)
    {
      auto u = vars.template value<uIdx>(evaluators);
      if (useNitsche)
      {
        auto ux = vars.template derivative<uIdx>(evaluators);
        nitsche.setBoundaryData(xi,penalty,u,uDirichletBoundaryValue,ux,unitMatrix<Scalar,dim>());
      }
      else
        dirichlet.setBoundaryData(penalty,u,uDirichletBoundaryValue);
    }

    Scalar d0() const
    {
      if (useNitsche)
        return nitsche.d0();
      else
        return dirichlet.d0();
    }

    template<int row>
    Dune::FieldVector<Scalar,1> d1(Kaskade::VariationalArg<Scalar,dim> const& argT) const
    {
      if (useNitsche)
        return nitsche.d1(argT);
      else
        return dirichlet.d1(argT);
    }

    template<int row, int col>
    Dune::FieldMatrix<Scalar,1,1> d2(Kaskade::VariationalArg<Scalar,dim> const &argT,
                                     Kaskade::VariationalArg<Scalar,dim> const &argA) const
    {
      if (useNitsche)
        return nitsche.d2(argT,argA);
      else
        return dirichlet.d2(argT,argA);
    }

  private:
    typename AnsatzVars::VariableSet const& vars;
    Scalar penalty;
    Dune::FieldVector<Scalar,1> uDirichletBoundaryValue;
    DirichletPenaltyBoundary<typename AnsatzVars::GridView,1> dirichlet;
    DirichletNitscheBoundary<typename AnsatzVars::GridView,1> nitsche;
    bool useNitsche;
  };

  template <int row>
  struct D1: public Kaskade::FunctionalBase<Kaskade::VariationalFunctional>::D1<row>
  {
    static bool const present   = true;
    static bool const constant  = false;

  };

  template <int row, int col>
  struct D2: public Kaskade::FunctionalBase<Kaskade::VariationalFunctional>::D2<row,col>
  {
    static bool const present = true;
    static bool const symmetric = true;
    static bool const lumped = false;
  };

  HeatFunctional(double gamma=1e6)
  : penalty(gamma)
  {}

  template <class Cell>
  int integrationOrder(Cell const& /* cell */, int shapeFunctionOrder, bool boundary) const
  {
    if (boundary)
      return 2*shapeFunctionOrder;
    else
    {
      int stiffnessMatrixIntegrationOrder = 2*(shapeFunctionOrder-1);
      int sourceTermIntegrationOrder = shapeFunctionOrder;        // as rhs f is constant, i.e. of order 0

      return std::max(stiffnessMatrixIntegrationOrder,sourceTermIntegrationOrder);
    }
  }

private:
  double penalty;
};
