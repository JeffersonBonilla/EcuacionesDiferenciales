# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Any
import sympy as sp
from sympy.solvers.ode import classify_ode
from sympy import dsolve, Function, Eq

app = FastAPI(title="EDO Solver API")

class SolveRequest(BaseModel):
    ode: str
    var: str = "x"
    function: str = "y"

class SolveResponse(BaseModel):
    status: str
    classification: List[str]
    is_homogeneous: bool
    homogeneity_explanation: str
    solution: str
    steps: List[str]
    raw_sympy: str

@app.get("/api/health")
def health():
    return {"status": "ok"}

def parse_ode(text, var_sym, func_sym):
    # Intent: try parse expressions like "dy/dx = ...", "y' + 2*x*y = sin(x)"
    # We attempt several parse strategies.
    x = var_sym
    y = func_sym(x)
    # Replace common notations to sympy friendly:
    txt = text.replace("dy/dx", "Derivative(y, x)").replace("y'", "Derivative(y, x)")
    # Provide local symbols
    local_dict = { 'x': x, 'y': y, 'Derivative': sp.Derivative }
    try:
        # Try to parse an equation
        expr = sp.sympify(txt, locals=local_dict)
        # If it's an equation object, keep it; else try to make Eq(expr,0)
        if isinstance(expr, sp.Equality):
            eq = expr
        else:
            # If expression contains Derivative(...) try to find LHS containing derivative
            # Fallback: Eq(expr,0)
            eq = Eq(expr, 0)
        return eq
    except Exception as e:
        # Last resort: try to split at '='
        if "=" in text:
            left, right = text.split("=", 1)
            try:
                left_s = sp.sympify(left, locals=local_dict)
                right_s = sp.sympify(right, locals=local_dict)
                return Eq(left_s, right_s)
            except Exception:
                pass
        raise e

def is_homogeneous_first_order(eq, x, yfun):
    # checks dy/dx = f(y/x) or can be rearranged to that form
    # This is a heuristic: create f = rhs / lhs etc.
    try:
        # try to get dy/dx = RHS
        if isinstance(eq, sp.Equality):
            lhs = eq.lhs
            rhs = eq.rhs
            # solve for Derivative(y,x)
            deriv = sp.Derivative(yfun(x), x)
            sol_for_deriv = sp.solve(Eq(lhs, rhs), deriv)
            if sol_for_deriv:
                f = sp.simplify(sol_for_deriv[0])
            else:
                # try if eq is Derivative(y,x) - f(x,y) = 0
                # move everything to rhs
                expr = sp.simplify(lhs - rhs)
                # isolate derivative
                expr = sp.solve(expr, deriv)
                f = expr[0] if expr else None
            if f is None:
                return False, "No se pudo aislar dy/dx"
            # check if f(x,y) can be expressed as g(y/x)
            t = sp.symbols('t')
            gx = sp.simplify(f.subs({x: x, yfun(x): t*x}))  # replace y by t*x => f(x, t*x)
            # if independent of x after substitution -> homogeneous of degree 0
            gx_s = sp.simplify(sp.factor(gx))
            free_symbols = gx_s.free_symbols
            if x in free_symbols:
                # still depends on x -> not homogeneous in this test
                return False, "Al sustituir y = t*x la expresión depende de x"
            else:
                return True, "Se puede escribir como f(y/x)"
        return False, "Ecuación no es igualdad"
    except Exception as e:
        return False, f"Error al analizar homogeneidad: {e}"

def generate_steps_for_first_order(eq, x, y):
    steps = []
    try:
        # classify via sympy
        methods = classify_ode(eq, y(x))
        steps.append(f"Clasificación SymPy: {methods}")
        # check separable
        if 'separable' in methods:
            steps.append("Método: Separable. Procedimiento:")
            # try to separate: dy/dx = g(x)*h(y)
            sol = dsolve(eq)
            steps.append("1) Reescriba como dy/dx = g(x) * h(y).")
            steps.append("2) Separe variables: dy/h(y) = g(x) dx.")
            steps.append("3) Integre ambos lados y + C.")
            steps.append(f"Solución (SymPy): {sp.pretty(sol)}")
            return steps
        if 'linear' in methods:
            steps.append("Método: Ecuación lineal de primer orden. Procedimiento:")
            steps.append("1) Reescribir en forma y' + P(x)*y = Q(x).")
            steps.append("2) Calcular factor integrante μ(x)=exp(∫P(x)dx).")
            steps.append("3) Multiplicar ecuación por μ(x) y simplificar -> (μ y)' = μ Q.")
            steps.append("4) Integrar: μ y = ∫ μ Q dx + C.")
            sol = dsolve(eq)
            steps.append(f"Solución (SymPy): {sp.pretty(sol)}")
            return steps
        if 'exact' in methods:
            steps.append("Método: Exacta. Procedimiento:")
            steps.append("1) Comprobar M dx + N dy = 0 con ∂M/∂y = ∂N/∂x.")
            steps.append("2) Encontrar función potencial Φ tal que Φ_x = M, Φ_y = N.")
            sol = dsolve(eq)
            steps.append(f"Solución (SymPy): {sp.pretty(sol)}")
            return steps
        # Fallback: return classification + sympy solution
        sol = dsolve(eq)
        steps.append("No se detectó un método simple implementado. Se devuelve la clasificación y la solución de SymPy.")
        steps.append(f"Solución (SymPy): {sp.pretty(sol)}")
        return steps
    except Exception as e:
        return [f"Error generando pasos automáticos: {e}"]

@app.post("/api/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    x = sp.symbols(req.var)
    y = Function(req.function)

    try:
        eq = parse_ode(req.ode, x, y)
    except Exception as e:
        return SolveResponse(
            status="error",
            classification=[],
            is_homogeneous=False,
            homogeneity_explanation=f"Error al parsear la ecuación: {e}",
            solution="",
            steps=[f"Error al parsear la ecuación: {e}"],
            raw_sympy=str(e)
        )

    # classification
    try:
        cls = classify_ode(eq, y(x))
    except Exception:
        cls = []
    # homogeneity check (heuristic for first-order)
    is_hom, expl = is_homogeneous_first_order(eq, x, y)
    # solution
    try:
        sol = dsolve(eq)
        sol_str = sp.srepr(sol)
    except Exception as e:
        sol_str = f"SymPy no pudo resolver: {e}"

    # steps (try to make detailed steps for first-order simple types)
    steps = generate_steps_for_first_order(eq, x, y)

    return SolveResponse(
        status="ok",
        classification=list(map(str, cls)),
        is_homogeneous=is_hom,
        homogeneity_explanation=expl,
        solution=str(sol),
        steps=steps,
        raw_sympy=sol_str
    )
