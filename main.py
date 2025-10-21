from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import sympy as sp
from sympy import Function, Eq, dsolve
from sympy.solvers.ode import classify_ode

app = FastAPI(
    title="EDO Solver API",
    version="1.0",
    description="API para resolver ecuaciones diferenciales ordinarias simbólicamente usando SymPy."
)

# ---------- MODELOS ----------
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
    latex_solution: str
    steps: List[str]
    latex_steps: List[str]

# ---------- UTILIDADES ----------
def parse_ode(text, var_sym, func_sym):
    x = var_sym
    y = func_sym(x)
    txt = (
        text.replace("dy/dx", "Derivative(y, x)")
            .replace("y'", "Derivative(y, x)")
            .replace("^", "**")
    )
    local_dict = {"x": x, "y": y, "Derivative": sp.Derivative, "sin": sp.sin, "cos": sp.cos, "exp": sp.exp}
    try:
        expr = sp.sympify(txt, locals=local_dict)
        if isinstance(expr, sp.Equality):
            return expr
        return Eq(expr, 0)
    except Exception:
        if "=" in text:
            left, right = text.split("=", 1)
            left_s = sp.sympify(left, locals=local_dict)
            right_s = sp.sympify(right, locals=local_dict)
            return Eq(left_s, right_s)
        raise ValueError("No se pudo interpretar la ecuación diferencial")

def is_homogeneous_first_order(eq, x, yfun):
    try:
        if isinstance(eq, sp.Equality):
            deriv = sp.Derivative(yfun(x), x)
            sol_for_deriv = sp.solve(eq, deriv)
            if not sol_for_deriv:
                return False, "No se pudo aislar dy/dx"
            f = sp.simplify(sol_for_deriv[0])
            t = sp.Symbol("t")
            gx = sp.simplify(f.subs({yfun(x): t * x}))
            if x in gx.free_symbols:
                return False, "Depende de x después de sustituir y = tx"
            else:
                return True, "Se puede expresar como f(y/x)"
        return False, "Ecuación no es igualdad"
    except Exception as e:
        return False, f"Error al analizar homogeneidad: {e}"

def generate_steps(eq, x, y):
    steps, latex_steps = [], []
    try:
        methods = classify_ode(eq, y(x))
        steps.append(f"Clasificación SymPy: {methods}")
        latex_steps.append(f"\\textbf{{Clasificación SymPy:}}\\ {methods}")

        if "separable" in methods:
            sol = dsolve(eq)
            steps += [
                "Método: Separable",
                "1) Escriba en forma dy/dx = g(x)h(y)",
                "2) Separe variables: dy/h(y) = g(x) dx",
                "3) Integre ambos lados",
                f"Solución: {sol}"
            ]
            latex_steps += [
                "\\text{Método: Separable}",
                "1) \\text{Escriba en forma } \\frac{dy}{dx} = g(x)h(y)",
                "2) \\text{Separe variables: } \\frac{dy}{h(y)} = g(x)\\,dx",
                "3) \\text{Integre ambos lados}",
                f"\\textbf{{Solución:}}\\ {sp.latex(sol)}"
            ]
            return steps, latex_steps

        if "linear" in methods:
            sol = dsolve(eq)
            steps += [
                "Método: Lineal de primer orden",
                "1) Forma estándar: y' + P(x)y = Q(x)",
                "2) Calcular μ(x) = e^{∫P(x)dx}",
                "3) Multiplicar la ecuación por μ(x)",
                "4) Integrar ambos lados",
                f"Solución: {sol}"
            ]
            latex_steps += [
                "\\text{Método: Lineal de primer orden}",
                "1) \\text{Forma estándar: } y' + P(x)y = Q(x)",
                "2) \\mu(x) = e^{\\int P(x)dx}",
                "3) \\text{Multiplicar por } \\mu(x)",
                "4) \\text{Integrar ambos lados}",
                f"\\textbf{{Solución:}}\\ {sp.latex(sol)}"
            ]
            return steps, latex_steps

        if "exact" in methods:
            sol = dsolve(eq)
            steps += [
                "Método: Exacta",
                "1) Verificar ∂M/∂y = ∂N/∂x",
                "2) Encontrar Φ tal que Φ_x = M, Φ_y = N",
                f"Solución: {sol}"
            ]
            latex_steps += [
                "\\text{Método: Exacta}",
                "1) \\text{Verificar } \\frac{\\partial M}{\\partial y} = \\frac{\\partial N}{\\partial x}",
                "2) \\text{Encontrar } \\Phi\\ \\text{tal que } \\Phi_x = M, \\Phi_y = N",
                f"\\textbf{{Solución:}}\\ {sp.latex(sol)}"
            ]
            return steps, latex_steps

        sol = dsolve(eq)
        steps.append("No se detectó un método simple. Se usa dsolve().")
        latex_steps.append("\\text{No se detectó un método simple. Se usa dsolve().}")
        latex_steps.append(f"\\textbf{{Solución:}}\\ {sp.latex(sol)}")
        return steps, latex_steps

    except Exception as e:
        return [f"Error al generar pasos: {e}"], [f"\\text{{Error al generar pasos: {e}}}"]

# ---------- ENDPOINTS ----------
@app.get("/api/health")
def health():
    return {"status": "ok", "message": "EDO Solver API activa"}

@app.post("/api/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    x = sp.Symbol(req.var)
    y = Function(req.function)
    try:
        eq = parse_ode(req.ode, x, y)
    except Exception as e:
        return SolveResponse(
            status="error",
            classification=[],
            is_homogeneous=False,
            homogeneity_explanation=str(e),
            solution="",
            latex_solution="",
            steps=[str(e)],
            latex_steps=[f"\\text{{Error: {e}}}"]
        )

    classification = [str(c) for c in classify_ode(eq, y(x))]
    is_hom, expl = is_homogeneous_first_order(eq, x, y)
    try:
        sol = dsolve(eq)
        sol_str = str(sol)
        sol_latex = sp.latex(sol)
    except Exception as e:
        sol_str = f"Error resolviendo: {e}"
        sol_latex = f"\\text{{Error resolviendo: {e}}}"

    steps, latex_steps = generate_steps(eq, x, y)
    return SolveResponse(
        status="ok",
        classification=classification,
        is_homogeneous=is_hom,
        homogeneity_explanation=expl,
        solution=sol_str,
        latex_solution=sol_latex,
        steps=steps,
        latex_steps=latex_steps
    )
