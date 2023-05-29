#=
Parcial 4 Análisis Numérico 2 Pablo Buitrago Jaramillo
=#
using Logging, ProgressBars
using LinearSolve, Cubature
using PyCall
fenics = pyimport("ffc")
using Plots

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
    
    Uex(x) = x^2*(1-x)^3
    Uexd(x) = 2x*(1-x)^3 - 3x^2*(1-x)^2
    Uexdd(x) = -20x^3+36x^2-18x+2
    
    #Uex(x) = x*(1-x)
    #Uexd(x) = 1-2x
    #Uexdd(x) = -2
    
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
        #---- Calculate the error MSE----
        @info "Calculating the error using MSE"
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
    # Plot the results in 4 subplots
    @info "Plotting the results"
    plot_array = []
    for n in N
        push!(plot_array,plot(Results[n][1],Results[n][2],label="Exact Solution",title="n=$n"))
        plot!(Results[n][1],Results[n][3],label="Numerical Solution")
    end
    plot(plot_array...,layout=(2,2),size=(800,800))
end