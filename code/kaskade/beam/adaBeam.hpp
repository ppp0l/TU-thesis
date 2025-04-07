
#ifndef BEAM_HH
#define BEAM_HH

#include <type_traits>
#include "fem/functional_aux.hh"
#include "fem/diffops/elastoVariationalFunctionals.hh"

using namespace boost::fusion;

// Function for listing the available materials in Kaskade
void printMaterial(const std::map<std::string, ElasticModulus>& map)
{
	std::cout<<"List of known materials:"<<std::endl;
	std::cout<<std::setw(20) <<"Material"<<std::setw(20) <<"Young's modulus"<<std::setw(20) <<"Poisson's ratio"<<std::endl<<std::endl;

    for (const auto& [key, value] : map) {
        std::cout <<std::setw(20)<<key<<std::setw(20)<< value.young()<<std::setw(20) <<value.poisson() <<std::endl;
    }
    std::cout << "\n";
}


// Deriving from FunctionalBase introduces default D1 and D2 structures.
template <class VarSet, class StrainTensor>
class ElasticityFunctional: public Kaskade::FunctionalBase<VariationalFunctional>
{
public:
  using Scalar = double;
  using AnsatzVars = VarSet;
  using TestVars = VarSet;
  using OriginVars = VarSet;
  static int const dim = AnsatzVars::Grid::dimension;
  using Vector = Dune::FieldVector<Scalar, dim>;
  using Matrix = Dune::FieldMatrix<Scalar, dim, dim>;
  static int constexpr u_Idx = 0;
  static int constexpr u_Space_Idx = result_of::value_at_c<typename AnsatzVars::Variables, u_Idx>::type::spaceIndex;
  
  using MaterialLaw = MaterialLaws::StVenantKirchhoff<dim,Scalar>;
  using ElasticEnergy = HyperelasticVariationalFunctional<MaterialLaw,StrainTensor>;

  class DomainCache 
  {
  public:
    DomainCache(ElasticityFunctional const& functional, typename AnsatzVars::VariableSet const& vars_, int flags=7)
    : vars(vars_), energy(functional.moduli)
    {}

    template <class Entity>
    void moveTo(Entity const& entity) {}

    template <class Position, class Evaluators>
    void evaluateAt(Position const& x, Evaluators const& evaluators)
    {
      using namespace boost::fusion;
      energy.setLinearizationPoint( at_c<u_Idx>(vars.data).derivative(at_c<u_Space_Idx>(evaluators)) );
    }

    Scalar d0() const
    {
      return energy.d0();
    }

    template<int row>
    Vector d1 (VariationalArg<Scalar,dim> const& arg) const
    {
      return energy.d1(arg);
    }

    template<int row, int col>
    Matrix d2 (VariationalArg<Scalar,dim> const& argTest, VariationalArg<Scalar,dim> const& argAnsatz) const
    {
      return energy.d2(argTest,argAnsatz);
    }

  private:
    typename AnsatzVars::VariableSet const& vars;
    ElasticEnergy energy;
  };

  class BoundaryCache : public CacheBase<ElasticityFunctional,BoundaryCache>
  {
  public:
    using FaceIterator = typename AnsatzVars::Grid::LeafIntersectionIterator;

    BoundaryCache(ElasticityFunctional const& f_, typename AnsatzVars::VariableSet const& vars_, int flags=7)
    : vars(vars_)
    {}

    void moveTo(FaceIterator const& face)
    {
      faceIt = &face;
    }

    template <class Evaluators>
    void evaluateAt(Dune::FieldVector<typename AnsatzVars::Grid::ctype,dim-1> const& x, Evaluators const& evaluators)
    {
      u0 = 0;
      u = at_c<u_Idx>(vars.data).value(at_c<u_Space_Idx>(evaluators));
      
      Vector left(0.); left[0] = -1.;                   // unit left pointing vector
      Vector up(0.); up[dim-1] = 1.;                    // unit upward pointing vector
      auto n = (*faceIt)->centerUnitOuterNormal();      // unit outer normal of local face
      
      Vector force(0.); force[dim-1] = -1.e8;
      
      if ( (n*left > 0.95) or (n*left < -0.95) )        // clamp beam on the left and right
      {
        alpha = 1e16;
        beta  = 0.;
      }
      else if( n*up > 0.95 )       // apply downward force on top
      {
        alpha = 0.;
        beta  = force;
      }
      else                        // homogeneous Neumann BCs on the remaining boundary
      {
        alpha = 0.;
        beta  = 0.;
      }
    }

    Scalar
    d0() const
    {
      return alpha*((u-u0)*(u-u0))/2 - beta*u;
    }

    template<int row>
    Scalar d1_impl (VariationalArg<Scalar,dim,dim> const& arg) const
    {
      return alpha*((u-u0)*arg.value) - beta*arg.value;
    }

    template<int row, int col>
    Scalar d2_impl (VariationalArg<Scalar,dim,dim> const &arg1, VariationalArg<Scalar,dim,dim> const &arg2) const
    {
      return alpha*(arg1.value*arg2.value);
    }

  private:
    typename AnsatzVars::VariableSet const& vars;
    Vector u, u0, beta;
    Scalar alpha;
    FaceIterator const* faceIt;
  };

  ElasticityFunctional(Kaskade::ElasticModulus const& moduli_): moduli(moduli_)
  {
  }


  template <class Cell>
  int integrationOrder(Cell const& /* cell */, int shapeFunctionOrder, bool boundary) const
  {
    if (boundary)
      return 2*shapeFunctionOrder;      // mass term u*u on boundary
    else
      return 2*(shapeFunctionOrder-1);  // energy term "u_x * u_x" in interior
  }

  Kaskade::ElasticModulus moduli;
};

#endif /* BEAM_HH */

