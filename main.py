from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sympy import symbols, Eq, Function, Derivative, dsolve, exp, integrate
from sympy.parsing.sympy_parser import parse_expr
import sympy as sp
import re

app = FastAPI(title="EcuSolver API")

# Habilitar CORS para Android
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Permitir todas las IPs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables simbólicas globales
x = symbols('x')
y = Function('y')

class SolveRequest(BaseModel):
    equation: str

# Funcion auxiliar: resolver ecuaciones lineales

def solve_linear_detailed(req: SolveRequest):
    print(f"DEBUG: Recibida ecuación: {req.equation}")
    try:

        original_eq_latex = (
            req.equation
            .replace("dy/dx", r"\frac{dy}{dx}")
            .replace("*", "")
            .replace("=", " = ")
        )

        # Normalizar para SymPy
        expr_str = req.equation.strip()
        expr_str = re.sub(r'\by\b', 'y(x)', expr_str)
        expr_str = (
            expr_str.replace("dy/dx", "Derivative(y(x), x)")
            .replace("y'", "Derivative(y(x), x)")
            .replace("y''", "Derivative(y(x), (x,2))")
            .replace("^", "**")
        )

        if "=" in expr_str:
            left, right = expr_str.split("=", 1)
            expr_str = f"({left.strip()}) - ({right.strip()})"

        local_dict = {'x': x, 'y': y, 'Derivative': Derivative}
        expr = parse_expr(expr_str, local_dict=local_dict)
        eq = Eq(expr, 0)
        print(f"DEBUG: Ecuación SymPy: {eq}")

        # Resolver con SymPy
        sol = dsolve(eq)
        if sol is None:
            raise ValueError("SymPy no pudo resolver la ecuación.")
        print(f"DEBUG: Solución cruda: {sol}")

        # Simplificar solucion
        rhs = sp.cancel(sol.rhs)
        rhs = sp.expand(rhs)
        sol = Eq(sol.lhs, rhs)
        print(f"DEBUG: Solución simplificada: {sol}")

        # Coeficientes de la ecuacion lineal: dy/dx + P(x)y = Q(x)
        dy_dx = Derivative(y(x), x)
        a_coeff = expr.coeff(dy_dx)
        b_coeff = expr.coeff(y(x))
        c_coeff = expr - a_coeff * dy_dx - b_coeff * y(x)

        if a_coeff != 0:
            P = b_coeff / a_coeff
            Q = -c_coeff / a_coeff
        else:
            P = 0
            Q = -expr / b_coeff

        # Factor integrante
        integral_P = integrate(P, x)
        mu = exp(integral_P)

        # Pasos en formato LaTeX
        steps = [
            f"Ecuación original: $${original_eq_latex}$$",
            f"Forma estándar: $$\\frac{{dy}}{{dx}} + ({sp.latex(P)})y = {sp.latex(Q)}$$",
            f"Factor integrante ($\\mu(x)$): $$e^{{\\int {sp.latex(P)} dx}} = e^{{{sp.latex(integral_P)}}} = {sp.latex(mu)}$$",
            f"Multiplicamos por $\\mu(x)$: $${sp.latex(mu)}\\frac{{dy}}{{dx}} + {sp.latex(mu)}({sp.latex(P)})y = {sp.latex(mu)}({sp.latex(Q)})$$",
            f"Integración: $${sp.latex(mu)}y = \\int {sp.latex(mu)}({sp.latex(Q)})\\,dx$$",
            f"Solución general: $${sp.latex(sol)}$$"
        ]

        return {"steps": steps}

    except Exception as e:
        print(f"DEBUG: Error en solve_linear_detailed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error al resolver: {str(e)}")

# Funcion auxiliar: resolver homogenea de primer orden

def solve_homogeneous_first_detailed(req: SolveRequest):
    print(f"DEBUG: Recibida ecuación: {req.equation}")
    try:
        original_eq_latex = (
            req.equation
            .replace("dy/dx", r"\frac{dy}{dx}")
            .replace("*", "")
            .replace("=", " = ")
        )

        # Normalizar para SymPy
        expr_str = req.equation.strip()
        expr_str = re.sub(r'\by\b', 'y(x)', expr_str)
        expr_str = expr_str.replace("dy/dx", "Derivative(y(x), x)").replace("^", "**")

        if "=" in expr_str:
            left, right = expr_str.split("=", 1)
            expr_str = f"({left.strip()}) - ({right.strip()})"

        local_dict = {'x': x, 'y': y, 'Derivative': Derivative}
        expr = parse_expr(expr_str, local_dict=local_dict)
        eq = Eq(expr, 0)
        print(f"DEBUG: Ecuación SymPy: {eq}")

        # Resolver con SymPy
        sol = dsolve(eq)
        if sol is None:
            raise ValueError("SymPy no pudo resolver la ecuación.")
        print(f"DEBUG: Solución cruda: {sol}")

        # Simplificar
        rhs = sp.cancel(sol.rhs)
        rhs = sp.expand(rhs)
        sol = Eq(sol.lhs, rhs)
        print(f"DEBUG: Solución simplificada: {sol}")

        # Pasos para homogenea de primer orden (cambio de variable v = y/x)
        steps = [
            f"Ecuación original: $${original_eq_latex}$$",
            f"Forma homogénea: $$\\frac{{dy}}{{dx}} = f\\left(\\frac{{y}}{{x}}\\right)$$",
            f"Cambio de variable: $$v = \\frac{{y}}{{x}}$$ entonces $$y = v x$$ y $$\\frac{{dy}}{{dx}} = v + x \\frac{{dv}}{{dx}}$$",
            f"Sustituyendo: $$v + x \\frac{{dv}}{{dx}} = f(v)$$",
            f"Separando variables: $$\\frac{{dv}}{{f(v) - v}} = \\frac{{dx}}{{x}}$$",
            f"Integrando: $$\\int \\frac{{dv}}{{f(v) - v}} = \\int \\frac{{dx}}{{x}}$$",
            f"Solución general: $${sp.latex(sol)}$$"
        ]

        return {"steps": steps}

    except Exception as e:
        print(f"DEBUG: Error en solve_homogeneous_first_detailed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error al resolver: {str(e)}")


# Funcion: resolver homogenea de segundo orden (con pasos)

def solve_homogeneous_second_detailed(req: SolveRequest):
    print(f"DEBUG: Recibida ecuación: {req.equation}")
    try:
        r = symbols('r') 

        original_eq_latex = (
            req.equation
            .replace("d²y/dx²", r"\frac{d^2 y}{dx^2}")
            .replace("dy/dx", r"\frac{dy}{dx}")
            .replace("*", "")
            .replace("=", " = ")
        )

        # Normalizar para SymPy
        expr_str = req.equation.strip()
        expr_str = re.sub(r'\by\b', 'y(x)', expr_str)
        expr_str = (
            expr_str.replace("d²y/dx²", "Derivative(y(x), (x,2))")
            .replace("dy/dx", "Derivative(y(x), x)")
            .replace("^", "**")
        )

        if "=" in expr_str:
            left, right = expr_str.split("=", 1)
            expr_str = f"({left.strip()}) - ({right.strip()})"

        local_dict = {'x': x, 'y': y, 'Derivative': Derivative}
        expr = parse_expr(expr_str, local_dict=local_dict)
        eq = Eq(expr, 0)
        print(f"DEBUG: Ecuación SymPy: {eq}")

        # Resolver con SymPy
        sol = dsolve(eq)
        if sol is None:
            raise ValueError("SymPy no pudo resolver la ecuación.")
        print(f"DEBUG: Solución cruda: {sol}")

        # Simplificar
        rhs = sp.cancel(sol.rhs)
        rhs = sp.expand(rhs)
        sol = Eq(sol.lhs, rhs)
        print(f"DEBUG: Solución simplificada: {sol}")

        # Pasos para homogenea
        steps = [
            f"Ecuación original: $${original_eq_latex}$$",
            f"Forma estándar: $$y'' + P(x) y' + Q(x) y = 0$$",
            f"Asumimos solución: $$y = e^{{{sp.latex(r)} x}}$$",
            f"Sustituyendo: $${sp.latex(r)}^2 e^{{{sp.latex(r)} x}} + P(x) {sp.latex(r)} e^{{{sp.latex(r)} x}} + Q(x) e^{{{sp.latex(r)} x}} = 0$$",
            f"Ecuación característica: $${sp.latex(r)}^2 + P(x) {sp.latex(r)} + Q(x) = 0$$",
            f"Resolviendo la ecuación característica...",
            f"Solución general: $${sp.latex(sol)}$$"
        ]

        return {"steps": steps}

    except Exception as e:
        print(f"DEBUG: Error en solve_homogeneous_second_detailed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error al resolver: {str(e)}")

# auxiliar: resolver ecuación general (cualquier tipo)
def solve_equation_general(req: SolveRequest, eq_type: str = "general"):
    print(f"DEBUG: Recibida ecuación: {req.equation}")
    try:
        expr_str = req.equation.strip()
        expr_str = re.sub(r'\by\b', 'y(x)', expr_str)
        expr_str = (
            expr_str.replace("dy/dx", "Derivative(y(x), x)")
            .replace("y'", "Derivative(y(x), x)")
            .replace("y''", "Derivative(y(x), (x,2))")
            .replace("^", "**")
        )

        if "=" in expr_str:
            left, right = expr_str.split("=", 1)
            expr_str = f"({left.strip()}) - ({right.strip()})"

        local_dict = {'x': x, 'y': y, 'Derivative': Derivative}
        expr = parse_expr(expr_str, local_dict=local_dict)
        eq = Eq(expr, 0)
        print(f"DEBUG: Ecuación SymPy: {eq}")

        sol = dsolve(eq)
        if sol is None:
            raise ValueError("SymPy no pudo resolver la ecuación.")
        print(f"DEBUG: Solución cruda: {sol}")

        rhs = sp.cancel(sol.rhs)
        rhs = sp.expand(rhs)
        sol = Eq(sol.lhs, rhs)
        print(f"DEBUG: Solución simplificada: {sol}")

        steps = [
            f"\\text{{{eq_type.capitalize()}: }} {sp.latex(eq)}",
            f"\\text{{Reescritura (lado izquierdo = 0): }} {sp.latex(expr)} = 0",
            f"\\text{{Solución general: }} {sp.latex(sol)}"
        ]

        return {"steps": steps}

    except Exception as e:
        print(f"DEBUG: Error en solve_equation_general: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error al resolver: {str(e)}")

# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------
@app.post("/solve_linear")
def solve_linear(req: SolveRequest):
    return solve_linear_detailed(req)

@app.post("/solve_homogeneous_first")
def solve_homogeneous_first(req: SolveRequest):
    return solve_homogeneous_first_detailed(req)

@app.post("/solve_homogeneous_second")
def solve_homogeneous_second(req: SolveRequest):
    return solve_homogeneous_second_detailed(req)

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
