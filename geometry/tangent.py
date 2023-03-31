from purias_utils.geometry.topology import *
from purias_utils.geometry.utils import differentiate


class SmoothCurveSet:
    """
    Maps specifically [0, 1] to R^{Cxd}, where C is the number of curves in the set.
    Calling this gamma and gamma_dot because of the Part III course.
    gamma_dot is the derivative of the curve at that input in [0, 1].
    """

    def __init__(
        self, gamma: Callable[[Points], T], codomain_dim: int, num_curves: int
    ):
        self.num_curves = num_curves
        self.domain = UnitRightRectangle(1)
        self.codomain_dim = codomain_dim
        self._gamma = gamma

    def gamma(self, ps: Union[Points, T]):
        """
        Requires an input of C points (scalars in [0, 1])
        Evaluates the cth curve on the cth point provided,
            returning a set of c points of in Euclidean space of dimension codomain_dim

        TODO: REMOVE ALL DEPENDENCE ON Union[Points, T]
        """
        if isinstance(ps, T):
            ps = Points(ps)
        assert len(ps) == self.num_curves
        output = check_and_map(ps, self.domain, self._gamma)
        assert output.coords.shape == (self.num_curves, self.codomain_dim)
        return output

    def gamma_dot(self, ps: Union[Points, T]):
        # Remove only the 1 dimension that gamma used to map from [0,1]
        return differentiate(self.gamma, ps)


class BasisSetAtPoint(SmoothCurveSet):
    """
    See documentation in TangentField below about why this exists

    eval_point --> x(p)
    codomain --> x(U)

    eval_ts will be used in TangentField to evaluate the gamma_dot at the right place
    """

    def __init__(self, eval_point: Points, codomain: RightRectangle):
        assert eval_point.coords.shape == (1, codomain.dim)  # TODO: batch this somehow!
        self.eval_point = eval_point
        eval_ts = codomain.left + (
            (eval_point.coords - codomain.left) / (codomain.right - codomain.left)
        )
        self.eval_ts = Points(eval_ts.T)
        self.codomain_widths = codomain.right - codomain.left
        self.codomain_left = codomain.left
        super(BasisSetAtPoint, self).__init__(
            self._basis_gamma, codomain_dim=codomain.dim, num_curves=codomain.dim
        )

    def _basis_gamma(self, ps: Union[Points, T]):
        """
        Again please look at TangentField. 
        These should output the same vector (self.eval_point) d times, but this whole process is
            important for autograd.
        """
        if isinstance(ps, Points):
            ps = ps.coords
        output = self.eval_point.coords.clone().repeat(self.codomain_dim, 1)
        output = output - (output * torch.eye(self.codomain_dim).to(output.device))
        # TODO: batch this somehow!
        new_diag = (self.codomain_left + self.codomain_widths * ps.flatten())
        output += torch.diag(new_diag).to(output.device)
        return Points(output)

    def basis_tangent_vectors(self):
        return self.gamma_dot(self.eval_ts)


class TangentField:
    """
    This is a kinda redundant way of defining vector fields, but it
    allows us to take full advantage of torch autograd if we need it.

    Provide to this object:
        domain - supposed to be the codomain of a chart, i.e. x(U) in Claudi and Branco 2022
        vector_field - defines the tangent vector field in the domain, i.e. maps each point in the domain
            to a vector in the same dimension. This is called v_p in Claudi and Branco 2022

    This is the weird redundant part:
        When evaluated on a point in the domain (i.e. x(p)), this object will *generate*
            a BasisSetAtPoint(SmoothCurveSet) object, which has C = d curves, each of which span one dimension
            of the domain, and passes through that point in the domain

        The *derivative* of the ith curve of these is called f_i in Claudi and Branco 2022

        The curve set gamma_dot is then evaluated on a set of t \in [0, 1], with t[i] corresponding
            to the proportion of the way x(p) is in x(U) along the ith dimension
            - i.e. the t values that would gamma evaluate to be x(p)

        This set of derivatives is the basis set of x(U), and when passed through x^-1 defines the basis
            set of the tangent space at p, TpM

    The script cylinder.py will show this!
    """

    def __init__(self, vector_field: Callable[[Points], T], domain: RightRectangle):
        self.vector_field = vector_field
        self.domain = domain

    def evaluate_unweighted_basis_coordinates(self, points: Union[Points, T]):
        """
        Evaluate without differentiating, allowing future Jacobian calculation, namely
        when composed with an embedding

        As name suggests, these are just coordinates (e.g. in x(U)), not the directional vector

        For p points in, the output will a BatchedPoints object with shape [p, d, d]
            p because of the batch size
            first d because you need that many basis coordinates to define x(U)
            second d because the basis coordinates that define x(U) have that dimensionality
        """
        all_curve_evaluations = []
        for eval_point in points:
            # TODO: BATCH THIS AT THE BasisSetAtPoint LEVEL TOO!
            basis_set = BasisSetAtPoint(eval_point, codomain=self.domain)
            basis_coords = basis_set.gamma(basis_set.eval_ts.coords)
            all_curve_evaluations.append(basis_coords.coords)
            # eval_point.coords, basis_coords.coords
        return BatchedPoints(torch.stack(all_curve_evaluations, 0))

    def weight_jacobian(self, charted_points: Union[Points, T], embedded_basis_jacobian: Union[Points, T]):
        """
        TESTING
        TODO: EXPLAIN THE RESHAPE LINE!

        In final line:
            b = batch size
            m = manifold dimension (i.e. of charted_points)
            d = final dimension (i.e. of the function that is differentiated to give embedded_basis_jacobian)
        """
        if isinstance(charted_points, T):
            charted_points = Points(charted_points)
        if isinstance(embedded_basis_jacobian, Points):
            embedded_basis_jacobian = embedded_basis_jacobian.coords

        # In case the manifold is 1D, make the Jacobian the right shape
        embedded_basis_jacobian = embedded_basis_jacobian.reshape(
            *embedded_basis_jacobian.shape[:2], -1
        )

        tangent_vectors = self.vector_field(charted_points).coords

        return torch.einsum('bdm,bm->bd', embedded_basis_jacobian, tangent_vectors)


        
