from itertools import pairwise

import numpy as np

from numethods.roots.solution import RootResult


class RootBracketingError(ValueError):
    def __init__(self):
        super().__init__("f(xl) and f(xu) must have different signs!")


class ToleranceError(ValueError):
    def __init__(self):
        super().__init__("Tolerance 'rtol' must be higher than the machine precision.")


def bisection(
    f: callable,
    xl: float,
    xu: float,
    rtol: float = 1e-6,
    args: tuple = (),
    verbose: bool = False,
) -> RootResult:
    """Bisection method.

    Implements the bisection method for root finding. The method
    repeatedly bisects the bracketing interval [xl, xu] where f(xl) and
    f(xu) have opposite signs, narrowing down the location of the root
    until the specified tolerance is met.

    Parameters
    ----------
    f : callable
        The function for which the root is sought. Must accept a float
        as its first argument and return a float.
    xl : float
        Lower bound of the initial bracketing interval. Must satisfy
        ``f(xl) * f(xu) < 0``.
    xu : float
        Upper bound of the initial bracketing interval. Must satisfy
        ``f(xl) * f(xu) < 0``.
    rtol : float, optional
        Relative tolerance for convergence. Iteration stops when the
        relative approximate error falls below ``rtol``. Default is
        1e-6.
    args : tuple, optional
        Extra arguments passed to ``f``. Default is ``()``.
    verbose : bool, optional
        If ``True``, prints the iteration table with `xl`, `xu`, root
        estimate `xr`, and relative error `ea`at each step. Default is
        ``False``.

    Returns
    -------
    RootResult
        Object containing the converged solution and iteration history.

    Raises
    ------
    RootBracketingError
        If ``f(xl)`` and ``f(xu)`` have the same sign (no bracket).
    ToleranceError
        If ``rtol`` is smaller than machine precision.

    Notes
    -----
    At each bisection step the root estimate is the interval midpoint:

        xr = (xl + xu) / 2

    The relative approximate error is estimated as (Chapra 2021, Eq.
    5.3):

        εa = | (xu - xl) / (xu + xl) |

    References
    ----------
    Chapra, S. C.; Canale, R. P. Numerical Methods for Engineers. 8th
    ed. New York, NY: McGraw-Hill Education, 2021. ISBN
    978-1-260-23077-3.

    Examples
    --------
    Find the root of f(x) = x**2 - 4 in the interval [0, 3]:

    >>> from numethods.roots.bracketing_methods import bisection
    >>> f = lambda x: x**2 - 4
    >>> result = bisection(f, xl=0, xu=3)
    >>> abs(result.solution - 2.0) < 1e-5
    True
    """

    def get_relative_approximate_error(xl: float, xu: float) -> float:
        # Chapra 2021, Eq. 5.3 (error for bisection method, inclusive 1st iteration)
        if xu + xl != 0:
            return abs((xu - xl) / (xu + xl))
        else:
            return float("inf")

    result = RootResult()

    fl = f(xl, *args)
    fu = f(xu, *args)
    result.add_feval(2)
    if fl * fu > 0:
        raise RootBracketingError
    xr = (xl + xu) / 2

    if abs(rtol * xr) < np.finfo(float).eps:
        raise ToleranceError

    ea = get_relative_approximate_error(xl, xu)
    result.add_iter(sol=xr, ea=ea)
    if verbose:
        print("-" * 96)
        print(" " * 35 + "Bisection method iterations")
        print("-" * 96)
        print(f"{'i':>3} | {'xl':>20} | {'xu':>20} | {'xr':>20} | {'ea':>20}")
        print(f"{result.nit:3d} | {xl:20.16f} | {xu:20.16f} | {xr:20.16f} | {ea:20.16f}")
    while ea > rtol:
        xr_old = xr
        fr_old = f(xr_old, *args)
        result.add_feval()
        if fl * fr_old < 0:
            xu = xr_old
        elif fl * fr_old > 0:
            xl = xr_old
            fl = fr_old
        elif fl * fr_old == 0.0:
            break
        xr = (xl + xu) / 2
        ea = get_relative_approximate_error(xl, xu)
        result.add_iter(sol=xr, ea=ea)
        if verbose:
            print(f"{result.nit:3d} | {xl:20.16f} | {xu:20.16f} | {xr:20.16f} | {ea:20.16f}")

    result.success = True
    result.status = 0
    result.message = "Converged."

    if verbose:
        print(f"\nConverged solution: {xr}")
        print("-" * 96)
        print("")
    return result


def false_position(
    f: callable,
    xl: float,
    xu: float,
    rtol: float = 1e-6,
    args: tuple = (),
    max_stuck: int | None = 2,
    verbose: bool = False,
) -> RootResult:
    """Modified false position method (regula falsi).

    Implements the modified false position (regula falsi) method for
    root finding. The method uses linear interpolation between the
    bracket endpoints to estimate the root, replacing one of the bounds
    at each iteration based on the sign of the function at the new
    estimate. When ``max_stuck=None`` the modification is disabled and
    the function reduces to the classic false position method.

    Parameters
    ----------
    f : callable
        The function for which the root is sought. Must accept a float
        as its first argument and return a float.
    xl : float
        Lower bound of the initial bracketing interval. Must satisfy
        ``f(xl) * f(xu) < 0``.
    xu : float
        Upper bound of the initial bracketing interval. Must satisfy
        ``f(xl) * f(xu) < 0``.
    rtol : float, optional
        Relative tolerance for convergence. Iteration stops when the
        relative approximate error falls below ``rtol``. Default is
        1e-6.
    args : tuple, optional
        Extra arguments passed to ``f``. Default is ``()``.
    max_stuck : int or None, optional
        Maximum number of consecutive iterations allowed on the same
        bracket side before the function value at that side is halved.
        ``None`` disables the modification entirely, reducing the method
        to the classic false position (regula falsi). ``2`` reproduces
        the modified regula falsi behavior from Chapra (2021). Values
        ``<= 0`` raise ``ValueError``. Default is 2.
    verbose : bool, optional
        If ``True``, prints the iteration table with `xl`, `xu`, root
        estimate `xr`, and relative error `ea`at each step. Default is
        ``False``.

    Returns
    -------
    RootResult
        Object containing the converged solution and iteration history.

    Raises
    ------
    RootBracketingError
        If ``f(xl)`` and ``f(xu)`` have the same sign (no bracket).
    ToleranceError
        If ``rtol`` is smaller than machine precision.
    ValueError
        If ``max_stuck`` is not ``None`` and ``max_stuck <= 0``.

    Notes
    -----
    The root estimate at each iteration is obtained by linear
    interpolation between the bracket endpoints:

        xr = xu - f(xu) * (xl - xu) / (f(xl) - f(xu))

    For the first iteration the relative approximate error is estimated
    by the largest half-width of the bracket relative to the root
    estimate:

        εa = max(xu - xr, xr - xl) / xr

    For subsequent iterations the standard approximate relative error is
    used (Chapra 2021, Eq. 3.5):

        εa = | (xr_new - xr_old) / xr_new |

    To break stagnation when one bracket endpoint is never updated, the
    modification halves the function value at that stagnant side once
    ``max_stuck`` consecutive iterations have been spent there. This
    artificially shifts the interpolation point toward the stagnant
    bound, accelerating convergence. Setting ``max_stuck=2`` matches the
    modified regula falsi described in Chapra (2021); setting
    ``max_stuck=None`` disables the halving entirely, giving the classic
    method.

    References
    ----------
    Chapra, S. C.; Canale, R. P. Numerical Methods for Engineers. 8th
    ed. New York, NY: McGraw-Hill Education, 2021. ISBN
    978-1-260-23077-3.

    Examples
    --------
    Find the root of f(x) = x**2 - 4 in the interval [0, 3]:

    >>> from numethods.roots.bracketing_methods import false_position
    >>> f = lambda x: x**2 - 4
    >>> result = false_position(f, xl=0, xu=3)
    >>> abs(result.solution - 2.0) < 1e-5
    True
    """
    if max_stuck is not None and max_stuck <= 0:
        raise ValueError("max_stuck must be None or a positive integer.")

    def get_relative_approximate_error(xl: float, xu: float, xr: float) -> float:
        # My deduction, like Chapra 2021, Eq. 5.3 but adapted for false position method
        if xr != 0:
            return abs(max(xu - xr, xr - xl) / xr)
        else:
            return float("inf")

    result = RootResult()

    fl = f(xl, *args)
    fu = f(xu, *args)
    result.add_feval(2)
    if fl * fu > 0:
        raise RootBracketingError
    xr = xu - (fu * (xl - xu)) / (fl - fu)

    if abs(rtol * xr) < np.finfo(float).eps:
        raise ToleranceError

    iu = 0
    il = 0
    ea = get_relative_approximate_error(xl, xu, xr)
    result.add_iter(sol=xr, ea=ea)
    if verbose:
        print("-" * 96)
        print(" " * 32 + "False position method iterations")
        print("-" * 96)
        print(f"{'i':>3} | {'xl':>20} | {'xu':>20} | {'xr':>20} | {'ea':>20}")
        print(f"{result.nit:3d} | {xl:20.16f} | {xu:20.16f} | {xr:20.16f} | {ea:20.16f}")
    while ea > rtol:
        xr_old = xr
        fr_old = f(xr_old, *args)
        result.add_feval()
        if fl * fr_old < 0:
            xu = xr_old
            fu = fr_old
            iu = 0
            il += 1
            if max_stuck is not None and il >= max_stuck:
                fl /= 2
        elif fl * fr_old > 0:
            xl = xr_old
            fl = fr_old
            il = 0
            iu += 1
            if max_stuck is not None and iu >= max_stuck:
                fu /= 2
        elif fl * fr_old == 0:
            break
        xr = xu - (fu * (xl - xu)) / (fl - fu)
        ea = get_relative_approximate_error(xl, xu, xr)
        result.add_iter(sol=xr, ea=ea)
        if verbose:
            print(f"{result.nit:3d} | {xl:20.16f} | {xu:20.16f} | {xr:20.16f} | {ea:20.16f}")

    result.success = True
    result.status = 0
    result.message = "Converged."

    if verbose:
        print(f"\nConverged solution: {xr}")
        print("-" * 96)
        print("")
    return result


def incremental_search(
    f: callable,
    a: float,
    b: float,
    relative_step: float = 0.01,
    method: str = "bisection",
    rtol: float = 1e-6,
    args: tuple = (),
    verbose: bool = False,
):
    """Find the root of a function using the incremental search method.

    Implements the incremental search method for root finding. The
    method divides the interval [a, b] into subintervals of size
    determined by `relative_step` and checks for sign changes in `f` to
    identify brackets for root finding.

    Parameters
    ----------
    f : callable
        The function for which the root is sought. Must accept a float
        as its first argument and return a float.
    a : float
        Start of the interval to search for roots.
    b : float
        End of the interval to search for roots.
    relative_step : float, optional
        Relative step size for incrementing through the interval. The
        actual step size is calculated as `relative_step * (b - a)`.
        Default is 0.01.
    method : str, optional
        Root-finding method to apply on identified brackets. Must be
        either "bisection" or "false_position". Default is "bisection".
    rtol : float, optional
        Relative tolerance for convergence when applying the
        root-finding method on identified brackets. Default is 1e-6.
    args : tuple, optional
        Extra arguments passed to ``f``. Default is ``()``.
    verbose : bool, optional
        If ``True``, prints information about identified brackets and
        their roots. Default is ``False``.

    Returns
    -------
    list[RootResult]
        A list of RootResult objects containing the converged solutions
        and iteration history for each identified bracket.
    """
    available_methods = {
        "bisection": bisection,
        "false_position": false_position,
    }
    if method not in available_methods:
        raise ValueError(
            f"Method '{method}' is not supported. Available methods: {available_methods.keys()}"
        )

    if a >= b:
        raise ValueError("`a` must be less than `b`")

    delta = relative_step * (b - a)

    if abs(delta * a) < np.finfo(float).eps or abs(delta * b) < np.finfo(float).eps:
        raise ValueError(
            "Relative step 'relative_step' must be respect the machine precision."
        )

    candidates = [a + i * delta for i in range(int((b - a) / delta) + 1)]

    solution = []
    for xl, xu in pairwise(candidates):
        try:
            solution.append(
                available_methods[method](
                    f=f, xl=xl, xu=xu, rtol=rtol, args=args, verbose=verbose
                )
            )
            if verbose:
                print(f"Bracket found: [{xl}, {xu}] -> Root: {solution.solution}")
        except RootBracketingError:
            pass

    return solution


if __name__ == "__main__":

    def f(x: float) -> float:
        return (667.38 / x) * (1 - np.exp(-0.146843 * x)) - 40.0

    def manning(x: float) -> float:
        return (0.471405 * (20 * x) ** (5 / 3)) / ((20 + 2 * x) ** (2 / 3)) - 5

    def water_temperature(Ta: float, oxigen_saturation: float) -> float:
        return (
            -139.34411
            + 1.575701e5 / Ta
            - 6.642308e7 / Ta**2
            + 1.243800e10 / Ta**3
            - 8.621949e11 / Ta**4
            - np.log(oxigen_saturation)
        )

    def poly_1(x: float) -> float:
        return x**10 - 1

    def poly_2(x: float) -> float:
        return (
            (x + 6.8)
            * (x + 2.5)
            * (x + 1.4)
            * (x - 0.2)
            * (x - 1.3)
            * (x - 3.7)
            * (x - 5.2)
        )

    def first_example():
        parameters = {
            "f": f,
            "xl": 12,
            "xu": 16,
            "rtol": 1e-6,
            "verbose": True,
        }
        sol_bs = bisection(**parameters)
        sol_fp = false_position(**parameters)
        true_sol = 14.780208593679469

        print("-" * 96)
        print(f"True solution: {true_sol}")
        print(f"Relative tolerance: {parameters['rtol']}")
        print(
            f"Bisection: solution = {sol_bs.solution}, relative error = {abs((true_sol - sol_bs.solution) / true_sol)}"
        )
        print(
            f"False-position: solution = {sol_fp.solution}, relative error = {abs((true_sol - sol_fp.solution) / true_sol)}"
        )
        print("-" * 96)

    def second_example():
        parameters = {
            "f": manning,
            "xl": 0.0,
            "xu": 1.0,
            "rtol": 1e-6,
            "verbose": True,
        }
        bisection(**parameters)
        false_position(**parameters)

    def third_example():
        parameters = {
            "f": water_temperature,
            "xl": 273.15,
            "xu": 273.15 + 40.0,
            "rtol": 1e-3,
            "args": (8,),
            "verbose": True,
        }
        T_b = bisection(**parameters)
        T_f = false_position(**parameters)
        print(
            f"Temperature in Celsius: Bisection = {T_b.solution - 273.15}, False Position = {T_f.solution - 273.15}"
        )

        print("\n" + "=" * 96 + "\n")

        parameters = {
            "f": water_temperature,
            "xl": 273.15,
            "xu": 273.15 + 40.0,
            "rtol": 1e-3,
            "args": (10,),
            "verbose": True,
        }
        T_b = bisection(**parameters)
        T_f = false_position(**parameters)
        print(
            f"Temperature in Celsius: Bisection = {T_b.solution - 273.15}, False Position = {T_f.solution - 273.15}"
        )

        print("\n" + "=" * 96 + "\n")

        parameters = {
            "f": water_temperature,
            "xl": 273.15,
            "xu": 273.15 + 40.0,
            "rtol": 1e-3,
            "args": (12,),
            "verbose": True,
        }
        T_b = bisection(**parameters)
        T_f = false_position(**parameters)
        print(
            f"Temperature in Celsius: Bisection = {T_b.solution - 273.15}, False Position = {T_f.solution - 273.15}"
        )

    def fourth_example():
        parameters = {
            "f": poly_1,
            "xl": 0.0,
            "xu": 1.3,
            "rtol": 1e-4,
            "verbose": True,
        }
        bisection(**parameters)
        false_position(**parameters)

    def fifth_example():
        # parameters =
        incremental_search(
            f=poly_2,
            a=-8.0,
            b=8.0,
            relative_step=0.01,
            method="bisection",
            rtol=1e-4,
            verbose=True,
        )

    first_example()
    input("Press Enter to continue to the next example...")
    second_example()
    input("Press Enter to continue to the next example...")
    third_example()
    input("Press Enter to continue to the next example...")
    fourth_example()
    input("Press Enter to continue to the next example...")
    fifth_example()
