# Gram matrix constraints and the SHAKE formalism

## 1. Setup

Let R be a matrix whose columns are particle positions, and let L be an incidence matrix that maps positions to displacements. The columns of RL are then displacement vectors between particles — if R = [r₀ r₁ ⋯], then a column of RL might be r₀₁ = r₁ − r₀, and so on.

We express all bond-length and bond-angle constraints simultaneously as a Gram matrix equation:

G ≡ ½(RL)ᵀ(RL) = G₀

where G₀ is a constant matrix. Diagonal entries enforce ½∥r₀₁∥² = const (bond lengths), and off-diagonal entries enforce ½ r₀₁ ⋅ r₁₂ = const (bond angles).

## 2. Constrained update

Given unconstrained positions S at some later time, we seek constrained positions R′ satisfying G(R′) = G₀ while minimizing displacement from S:

min ½∥R′ − S∥²_F subject to G(R′) = G₀

The Lagrangian is:

ℒ = ½∥R′ − S∥²_F + tr(Λᵀ(G(R′) − G₀))

where Λ is a symmetric matrix of Lagrange multipliers — one per entry of the Gram matrix equation. Because the constraints are packaged as a matrix equation, the multipliers naturally form a matrix too.

## 3. The correction

Differentiating ℒ with respect to R′ and setting to zero:

R′ − S + R′LΛLᵀ = 0

This is an **implicit** equation for R′ — the correction direction involves R′L, the *new* constrained displacements, not the old ones. SHAKE resolves this by linearizing: replace R′ with R in the correction term:

**R′ ≈ S − RLΛLᵀ**

This linearization is what makes SHAKE an **iterative** method: solve for Λ using the old displacements RL, update positions, re-evaluate, and repeat until the constraint residual is sufficiently small.

## 4. Derivative of G with respect to R

The correction formula rests on the derivative ∂(½∥R′ − S∥²_F)/∂R′ + ∂f/∂R′ = 0, where f = tr(Λᵀ(½(R′L)ᵀ(R′L) − G₀)). We need ∂f/∂R′, which requires dG/dR.

The differential of G is:

dG = ½[Lᵀ dRᵀ RL + LᵀRᵀ dR L] = Sym(LᵀRᵀ dR L)

For a scalar f with S = ∂f/∂G symmetric, df = tr(S dG) gives:

df = ½ tr(SLᵀRᵀ dR L) + ½ tr(SLᵀ dRᵀ RL)

The second term transforms via tr(A dRᵀ B) = tr((BA)ᵀ dR):

df = ½ tr(RLSLᵀ dR) + ½ tr(RLSLᵀ dR) = tr(RLSLᵀ dR)

The two terms are equal — one from dR, one from dRᵀ — which cancels the factor of ½. The result is:

**∂f/∂R = RLSLᵀ**

Setting S = Λ recovers the correction RLΛLᵀ.

## 5. Summary

By packaging constraints as a Gram matrix equation G = G₀ rather than individual scalar equations, the multipliers form a symmetric matrix Λ, and the position correction factorizes cleanly as RLΛLᵀ through the product RL. The exact first-order condition is implicit in R′; the SHAKE linearization replaces R′L with RL, making the method iterative.