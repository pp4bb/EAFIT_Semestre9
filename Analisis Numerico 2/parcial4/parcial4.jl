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

# PUNTO 1.A SYMPY

"""
function punto1a()
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
    Uex = sin(4π*x)*cos(4π*x)
    f(x) = diff(Uex(x),x,2)
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
        # Joint Reference Functions
        λ(z) = Piecewise(
            (ϕ(Δx.subs(y,z)), z % 2 == 0),
            (ψ(Δx.subs(y,z)), z % 2 != 0)
        )
        #---- Derivative of the reference functions ----
        Vλ = [λ(z) for z in 1:xlen]
        Vλ1 = diff.(Vλ)
        #---- Assemble the matrix ----
        @info "Assembling the matrix and the vector"
        K = zeros(xlen,xlen)
        b = zeros(xlen)
        for i in tqdm(1:xlen)
            for j in i:xlen
                K[i,j] = integrate(Vλ1[i]*Vλ1[j],(x,0,1)).evalf()
                K[j,i] = K[i,j]
            end
            b[i] = integrate(-f(x)*Vλ[i],(x,0,1)).evalf()
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
        Uh = sum(alpha[j]*Vλ[j] for j in 1:xlen)
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
        xlabel!("x")
        ylabel!("u(x)")
    end
    curves = plot(plot_array...,layout=(2,2),size=(800,800))
    savefig(curves,"punto1a_curves.png")
end

"""

# PUNTO 1.B

"""
function punto1b()
    @vars x t real=true
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
    #---- Exact Solution and its derivatives ----
    Uex = sin(4π*x)*exp(-4π^2*t)
    Uexx1 = diff(Uex, x)
    Uexx2 = diff(Uex, x, 2)
    Uext1 = diff(Uex, t)
    f = Uext1 - Uexx2 + Uexx1
    #---- Parameters ----
    Δt = 0.1
    T = 1.0 # Final time
    N = [5]
    domain = range(0,stop=1,length=1000)
    Results = Dict()
    for n in N
        hx = 1/n # Espacial step size
        xlen = 2n-1 # Number of nodes
        tlen = Int(1/Δt+1) # Number of time steps
        xmesh = 0:hx/2:1 # Espacial Mesh
        @info "length(xmesh) = $xlen"
        # Change of variable, it is diferent for this point to help the computations
        Δx = (x-y.*(hx/2))/hx
        #---- Base Functions ----
        # Joint Reference Functions
        λ(z) = Piecewise(
            (ϕ(Δx.subs(y,z)), z % 2 == 0),
            (ψ(Δx.subs(y,z)), z % 2 != 0)
        )
        I = 1:xlen
        Vλ = [λ(z) for z in I]
        #=
        The matricial problem is:
        α'M + α(K+M) = b
        =#
        #---- Derivative of the Base Functions ----
        Vλ1 = diff.(Vλ)
        #---- Assemble the matrix and the vector----
        @info "Assembling the matrices and the vectors"
        K = zeros(xlen,xlen)
        M = zeros(xlen,xlen)
        for i in tqdm(1:xlen)
            for j in i:xlen
                K[i,j] = integrate(Vλ1[i]*Vλ1[j],(x,0,1)).evalf()
                K[j,i] = K[i,j]
                M[i,j] = integrate(Vλ[i]*Vλ[j],(x,0,1)).evalf()
                M[j,i] = M[i,j]
            end
        end
        #---- Calculate α(t)----
        U0 = Uex(0,x)
        @info "Calculating α(t)"
        intern_product = [integrate(U0*Vλ[i],(x,0,1)).evalf() for i in 1:xlen]
        alpha = zeros(xlen, Int(tlen))
        problem = LinearProblem(M, intern_product)
        linsolve = init(problem)
        alpha[:,1] = solve(linsolve, SVDFactorization())
        b = [integrate(f*Vλ[i],(x,0,1)).evalf() for i in 1:xlen]
        B(t) = [b[i](t) for i in 1:xlen]
        aux_matrix = M + Δt*K + Δt*M
        for i in tqdm(1:tlen-1)
            aux_vector = Δt*B(i*Δt) + M*alpha[:,i]
            problem = LinearProblem(aux_matrix, aux_vector)
            linsolve = init(problem)
            alpha[:,i+1] = solve(linsolve, SVDFactorization())
        end
        #---- Calculate the numerical solution ----
        @info "Calculating the numerical solution"
        Uh = transpose(Vλ)*alpha
        #---- Calculate the errors ----
        @info "Calculating the errors"
        errors_linf = zeros(tlen)
        tmesh = 0:Δt:T
        for i in 1:tlen
            Yex = Uex.(tmesh[i],domain)
            Yh = Uh[i].(domain)
            errors_linf[i] = maximum(abs.(Yex-Yh))
        end
        @info "Errors" errors_linf
        #---- Save the results ----
        Results[n] = [domain,tmesh,Uex,Uh,errors_linf]
    end
    #---- Plot the results ----
    @info "Plotting the results"
    for n in N
        idx = rand(1:length(Results[n][2]))
        t = Results[n][2][idx]
        Yex = Results[n][3].(t,Results[n][1])
        Yh = Results[n][4][idx].(Results[n][1])
        curves = plot(Results[n][1],Yex,label="Exact Solution", title="n = $n, t = $t")
        plot!(curves,Results[n][1],Yh,label="Numerical Solution")
        xlabel!("x")
        ylabel!("u(x,t)")
        savefig(curves, "Punto1b_curves_$n.png")
        curves = plot(Results[n][2],Results[n][5], title="Errors, n = $n")
        xlabel!("t")
        ylabel!("Error L∞")
        savefig(curves, "Punto1b_errors_$n.png")
    end
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
    Uex = sin(4π*x)*(1-cos(4π*x))
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
        #---- Base Functions ----
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
    savefig(errors,"punto2_errors.png")
    # Plot the results in 4 subplots
    @info "Plotting the results"
    plot_array = []
    for n in N
        push!(plot_array,plot(Results[n][1],Results[n][2],label="Exact Solution",title="n=$n"))
        plot!(Results[n][1],Results[n][3],label="Numerical Solution")
    end
    curves = plot(plot_array...,layout=(2,2),size=(800,800))
    savefig(curves,"punto2_curves.png")
end

############################## PUNTO 2 ########################################
"""
# PUNTO 3
"""
function punto3()
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
    Uex = sin(4π*x)*(1-cos(4π*x))
    Uex1 = diff(Uex,x)
    Uex2 = diff(Uex,x,2)
    Uex3 = diff(Uex,x,3)
    f(x) = Uex3 - Uex2
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
                aux1 = Vϕ1[j]*Vϕ1[i]
                aux2 = Vϕ2[j]*Vϕ1[i]
                K1[i,j] = integrate(aux1-aux2,(x,0,1)).evalf()
                aux1 = Vψ1[j]*Vϕ1[i]
                aux2 = Vψ2[j]*Vϕ1[i]
                M1[i,j] = integrate(aux1-aux2,(x,0,1)).evalf()
                aux1 = Vϕ1[j]*Vψ1[i]
                aux2 = Vϕ2[j]*Vψ1[i]
                K2[i,j] = integrate(aux1-aux2,(x,0,1)).evalf()
                aux1 = Vψ1[j]*Vψ1[i]
                aux2 = Vψ2[j]*Vψ1[i]
                M2[i,j] = integrate(Vψ2[j]*Vϕ1[i],(x,0,1)).evalf()
            end
            b1[i] = integrate(f(x)*Vϕ[i],(x,0,1)).evalf()
            b2[i] = integrate(f(x)*Vψ[i],(x,0,1)).evalf()
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
    savefig(errors,"punto3_errors.png")
    # Plot the results in 4 subplots
    @info "Plotting the results"
    plot_array = []
    for n in N
        push!(plot_array,plot(Results[n][1],Results[n][2],label="Exact Solution",title="n=$n"))
        plot!(Results[n][1],Results[n][3],label="Numerical Solution")
    end
    curves = plot(plot_array...,layout=(2,2),size=(800,800))
    savefig(curves,"punto3_curves.png")
end