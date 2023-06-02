#=
Parcial 4 Análisis Numérico 2 Pablo Buitrago Jaramillo
=#
using Logging, ProgressBars
using LinearSolve, Cubature
using BenchmarkTools    
using Plots
using SymPy
Piecewise = SymPy.sympy.Piecewise
diff = SymPy.sympy.diff
integrate = SymPy.sympy.integrate


############################## PUNTO 1 ########################################
"""

# PUNTO 1.A

"""
function punto1a()
    #---- Reference Functions ----
    #=
    ϕ = 
        ((1+x)*(1+2x), (-1 ≦ x) ∧(x ≦ 0)),
        ((1-x)*(1-2x), (0 ≦ x)∧ (x ≦ 1)),
    =#
    function ϕ(x)
        if abs(x) > 1
            return 0
        elseif x <= 0
            return (1+x)*(1+2x)
        else
            return (1-x)*(1-2x)
        end
    end

    function dϕ(x)
        if -1 <= x <= 0
            return 4x+3
        elseif 0 < x <= 1
            return 4x-3
        else
            return 0
        end
    end
    #=
    ψ = 
        (1-4x^2, abs(x) ≦ 1/2),
        (0, abs(x) > 1/2)
    =#
    ψ(x) = abs(x) <= 1/2 ? 1-4x^2 : 0
    function dψ(x)
        if abs(x) <= 1/2
            return -8x
        else
            return 0
        end
    end
    #---- Exact Solution ----
    
    #Uex(x) = x^2*(1-x)^3
    #Uexd(x) = 2x*(1-x)^3 - 3x^2*(1-x)^2
    #Uexdd(x) = -20x^3+36x^2-18x+2
    
    Uex(x) = x*(1-x)
    Uexd(x) = 1-2x
    Uexdd(x) = -2
    
    #---- Numerical Integration ----
    function quad_trap(f,a,b,N) 
        h = (b-a)/N
        int = h * ( f(a) + f(b) ) / 2
        for k=1:N-1
            xk = (b-a) * k/N + a
            int = int + h*f(xk)
        end
        return int
    end

    function quadgk(f, a, b)
        quad_trap(f,a,b,2^12)
    end

    #---- Numerical Solutions ----
    N = [5, 10, 50, 100]
    #N = [20]
    Results = Dict()
    for n ∈ N
        h = 1/n
        #---- Uniform mesh in [0,1] -----
        #X = range(0,stop=1,length=2n+1)
        X = 0+h/2:h/2:1-h/2
        xlen = length(X)
        exvals = [1, xlen]
        domain = range(0,stop=1,length=1000)
        Yex = Uex.(domain)
        #---- Join the test functions ----
        λ(x,j) = j ∈ exvals ? 0 : (j % 2 != 0 ? ϕ((x-X[j])/h) : ψ((x-X[j])/h))
        dλ(x,j) = j ∈ exvals ? 0 : (j % 2 != 0 ? dϕ((x-X[j])/h)*1/h : dψ((x-X[j])/h)*1/h)
        #---- Assemble the matrix ----
        @info "Assembling the matrix"
        A = zeros(xlen,xlen)
        for i in tqdm(1:xlen)
            for j in 1:xlen
                A[i,j] = quadgk(x->dλ(x,i)*dλ(x,j),BigFloat(0),BigFloat(1))[1]
            end
        end
        display(A)
        #---- Assemble the vector ----
        @info "Assembling the vector"
        b = zeros(xlen)
        for i in tqdm(1:xlen)
            b[i] = quadgk(x->-Uexdd(x)*λ(x,i),BigFloat(0),BigFloat(1))[1]
        end
        #---- Solve the system ----
        @info "Solving the system"
        #alpha = A\b
        problem = LinearProblem(A,b)
        linsolve = init(problem)
        alpha = solve(linsolve, SVDFactorization())
        display(alpha)
        #---- Calculate Numerical Solution Uh = ∑αjδj ----
        @info "Calculating the numerical solution"
        Uh(x) = sum(alpha[j]*λ(x,j) for j in 1:xlen)
        Yh = Uh.(domain)
        #---- Calculate the error Linf----
        @info "Calculating the error Linf"
        error = maximum(abs.(Yex-Yh))
        @info "Error: $error"
        #---- Save the results ----
        Results[n] = (domain,Yex,Yh,error)
        #=
        #plot δ functions
        plot()
        for j in 1:xlen
            plot!(domain,λ.(domain,j))
        end
        gui()
        =#
    end
    # Make a table with the errors for each n
    @info "Making a table with the errors for each n"
    table = zeros(length(N),2)
    for (i,n) in enumerate(N)
        table[i,1] = n
        table[i,2] = Results[n][4]
    end
    display(table)
    # Plot the errors
    @info "Plotting the errors"
    errors = plot()
    e = []
    for (i,n) in enumerate(N)
        push!(e,Results[n][4])
    end
    plot!(N,e,label="Error")
    savefig(errors,"punto1a_errors.png")
    # Plot the results in 4 subplots
    @info "Plotting the results"
    plot_array = []
    for n in N
        push!(plot_array,plot(Results[n][1],Results[n][2],label="Exact Solution",title="n=$n"))
        plot!(Results[n][1],Results[n][3],label="Numerical Solution")
    end
    curves = plot(plot_array...,layout=(2,2),size=(800,800))
    savefig(curves,"punto1a_curves.png")
end

"""

# PUNTO 1.A SYMPY

"""
function punto1a_sympy()
    @vars x real=true
    @vars y z positive=true integer=true
    #---- Reference Functions ----
    ϕ(x) = Piecewise(
        ((1+x)*(1+2x), (-1 ≦ x) ∧(x ≦ 0)),
        ((1-x)*(1-2x), (0 ≦ x)∧ (x ≦ 1)),
        (0, true)
    )
    ψ(x) = Piecewise(
        (1-4x^2, abs(x) ≦ 1/2),
        (0, true)
    )
    #---- Exact Solution ----
    Uex = x*(1-x)
    f(x) = diff(Uex(x),x,2)
    #---- Parameters ----
    N = [5, 10, 50, 100]
    #N = [10]
    domain = range(0,stop=1,length=1000)
    Results = Dict()
    for n in N
        h = 1/2n # Mesh size
        xlen = 2n-1 # Number of nodes
        mesh = 0+h:h:1-h # Mesh
        @info "length(mesh) = $(length(mesh))"
        Δx = (x-y.*h)/(2*h) # Change of variable
        # Joint Reference Functions
        λ(z) = Piecewise(
            (ϕ(Δx.subs(y,z)), z % 2 == 0),
            (ψ(Δx.subs(y,z)), z % 2 != 0)
        )
        #---- Derivative of the reference functions ----
        V = [λ(z) for z in 1:xlen]
        dV = diff.(V)
        #---- Assemble the matrix ----
        @info "Assembling the matrix and the vector"
        K = zeros(xlen,xlen)
        b = zeros(xlen)
        for i in tqdm(1:xlen)
            for j in 1:xlen
                K[i,j] = integrate(dV[i]*dV[j],(x,0,1)).evalf()
            end
            b[i] = integrate(-f(x)*V[i],(x,0,1)).evalf()
        end
        @info "K ="
        display(K)
        @info "b ="
        display(b)
        #---- Solve the system ----
        @info "Solving the system"
        #alpha = K\b
        problem = LinearProblem(K,b)
        linsolve = init(problem)
        alpha = solve(linsolve, SVDFactorization())
        display(alpha)
        #---- Calculate Numerical Solution Uh = ∑αjδj ----
        @info "Calculating the numerical solution"
        #Uh(x) = sum(alpha[j]*λ(x,j) for j in 1:xlen)
        Uh = sum(alpha[j]*V[j] for j in 1:xlen)
        Yh = Uh.(domain)
        #---- Calculate the exact solution ----
        Yex = Uex.(domain)
        #---- Calculate the error Linf----
        @info "Calculating the error using Linf"
        errorLinf = maximum(abs.(Yex-Yh))
        #---- Calculate the L2 error ----
        #@info "Calculating the error using L2"
        #errorL2 = integrate((Yex-Yh).^2,(x,0,1)).evalf()
        errorL2 = 0
        #---- Save the errors ----
        error = (errorLinf,errorL2)
        #---- Save the results ----
        Results[n] = (domain,Yex,Yh,error)
    end
    # Make a table with the errors for each n
    @info "Table with the errors Linf for each n"
    table = zeros(length(N),2)
    for (i,n) in enumerate(N)
        table[i,1] = n
        table[i,2] = Results[n][4][1]
    end
    display(table)
    #=
    @info "Table with the errors Linf for each n"
    table = zeros(length(N),2)
    for (i,n) in enumerate(N)
        table[i,1] = n
        table[i,2] = Results[n][4][2]
    end
    display(table)
    =#
    # Plot the errors
    @info "Plotting the errors"
    errors = plot()
    e1, e2 = [], []
    for (i,n) in enumerate(N)
        push!(e1,Results[n][4][1])
        #push!(e2,Results[n][4][2])
    end
    plot!(N,e1,label="Error Linf")
    #plot!(N,e2,label="Error L2")
    savefig(errors,"punto1aSym_errors.png")
    # Plot the results in 4 subplots
    @info "Plotting the results"
    plot_array = []
    for n in N
        push!(plot_array,plot(Results[n][1],Results[n][2],label="Exact Solution",title="n=$n"))
        plot!(Results[n][1],Results[n][3],label="Numerical Solution")
    end
    curves = plot(plot_array...,layout=(2,2),size=(800,800))
    savefig(curves,"punto1aSym_curves.png")
end

"""

# PUNTO 1.B

"""
function punto1b()

end


############################## PUNTO 2 ########################################
"""
# PUNTO 2
"""
function punto2()
    @vars x real=true
    @vars y z positive=true integer=true
    #---- Reference Functions ----
    ϕ(x) = Piecewise(
        ((1+x)^2*(1-2x), (-1 ≦ x) ∧(x ≦ 0)),
        ((1-x)^2*(1+2x), (0 ≦ x)∧ (x ≦ 1)),
        (0, true)
    )
    ψ(x) = Piecewise(
        (x*(1+x)^2, (-1 ≦ x) ∧(x ≦ 0)),
        (x*(1-x)^2, (0 ≦ x)∧ (x ≦ 1)),
        (0, true)
    )
    #---- Exact Solu0tion and its derivatives ----
    Uex = sin(4π*x)
    Uex1 = diff(Uex,x)
    Uex2 = diff(Uex,x,2)
    f(x) = diff(Uex,x,3)
    #---- Parameters ----
    #N = [5, 10, 50, 100]
    N = [5]
    domain = range(0,stop=1,length=1000)
    Results = Dict()
    for n in N
        h = 1/2n # Mesh size
        xlen = 2n-1 # Number of nodes
        mesh = 0+h:h:1-h # Mesh
        @info "length(mesh) = $(length(mesh))"
        Δx = (x-y.*h)/(2*h) # Change of variable
        # Base Functions
        ϕj(z) = ϕ(Δx.subs(y,z))
        ψj(z) = ψ(Δx.subs(y,z))
        I = 1:xlen
        Vϕ = [ϕj(z) for z in I]
        Vψ = [ψj(z) for z in I]
        #---- Derivative of the Base Functions ----
        Vϕ1 = diff.(Vϕ)
        Vϕ2 = diff.(Vϕ1)
        Vψ1 = diff.(Vψ)
        Vψ2 = diff.(Vψ1)
        #= 
        We have two matricial systems:
        α1K1 + M1 = b1
        α2K2 + M2 = b2
        =#
        #---- Assemble the matrices and the vectors----
        @info "Assembling the matrices and the vectors"
        K1 = zeros(xlen,xlen)
        K2 = zeros(xlen,xlen)
        M1 = zeros(xlen,xlen)
        M2 = zeros(xlen,xlen)
        b1 = zeros(xlen)
        b2 = zeros(xlen)
        for i in tqdm(1:xlen)
            for j in 1:xlen
                K1[i,j] = integrate(Vϕ2[j]*Vϕ1[i],(x,0,1)).evalf()
                M1[i,j] = integrate(Vψ2[j]*Vψ1[i],(x,0,1)).evalf()
                K2[i,j] = integrate(Vϕ2[j]*Vψ1[i],(x,0,1)).evalf()
                M2[i,j] = integrate(Vψ2[j]*Vϕ1[i],(x,0,1)).evalf()
            end
            b1[i] = integrate(-f(x)*Vϕ[i],(x,0,1)).evalf()
            b2[i] = integrate(-f(x)*Vψ[i],(x,0,1)).evalf()
        end
        #---- Solve the systems ----
        @info "Solving the systems"
        #alpha1
        problem1 = LinearProblem([K1 M1],b1)
        linsolve1 = init(problem1)
        alpha1 = solve(linsolve1, SVDFactorization())
        #alpha2
        problem2 = LinearProblem([K2 M2],b2)
        linsolve2 = init(problem2)
        alpha2 = solve(linsolve2, SVDFactorization())
        #---- Calculate the numerical solution ----
        @info "Calculating the numerical solution"
        Uh = sum(alpha1[j]*Vϕ[j] + alpha2[j]*Vψ[j] for j in 1:xlen)
        Yh = Uh.(domain)
        #---- Calculate the exact solution ----
        Yex = Uex.(domain)
        #---- Calculate the error Linf----
        @info "Calculating the error using Linf"
        errorLinf = maximum(abs.(Yex-Yh))
        #---- Calculate the L2 error ----
        @info "Calculating the error using L2"
        #errorL2 = integrate((Uex-Uh)^2,(x,0,1)).evalf()
        errorL2 = 0
        #---- Save the errors ----
        error = (errorLinf,errorL2)
        #---- Save the results ----
        Results[n] = (domain,Yex,Yh,error)
    end
    # Make a table with the errors for each n
    @info "Table with the errors Linf for each n"
    table = zeros(length(N),2)
    for (i,n) in enumerate(N)
        table[i,1] = n
        table[i,2] = Results[n][4][1]
    end
    display(table)
    #=
    @info "Table with the errors errorL2 for each n"
    table = zeros(length(N),2)
    for (i,n) in enumerate(N)
        table[i,1] = n
        table[i,2] = Results[n][4][2]
    end
    display(table)
    =#
    # Plot the errors
    @info "Plotting the errors"
    errors = plot()
    e1, e2 = [], []
    for (i,n) in enumerate(N)
        push!(e1,Results[n][4][1])
        #push!(e2,Results[n][4][2])
    end
    plot!(N,e1,label="Error Linf")
    #plot!(N,e2,label="Error L2")
    savefig(errors,"punto2Sym_errors.png")
    # Plot the results in 4 subplots
    @info "Plotting the results"
    plot_array = []
    for n in N
        push!(plot_array,plot(Results[n][1],Results[n][2],label="Exact Solution",title="n=$n"))
        plot!(Results[n][1],Results[n][3],label="Numerical Solution")
    end
    curves = plot(plot_array...,layout=(2,2),size=(800,800))
    savefig(curves,"punto2Sym_curves.png")
end

############################## PUNTO 2 ########################################
"""
# PUNTO 3
"""