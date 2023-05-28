#=
Parcial 4 Análisis Numérico 2 Pablo Buitrago Jaramillo
=#
using Logging, ProgressBars
using QuadGK, LinearSolve
using Plots

function punto1()
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
    
    #---- Numerical Solutions ----
    N = [5, 10, 50, 100]
    #N = [6]
    @info "" typeof(N)
    Results = Dict()
    for n ∈ N
        h = 1/n
        #---- Uniform mesh in [0,1] -----
        exvals = [1, 2n+1]
        X = range(0,stop=1,length=2n+1)
        domain = range(0,stop=1,length=1000)
        Yex = Uex.(domain)
        #---- Join the test functions ----
        λ(x,j) = j ∈ exvals ? 0 : (j % 2 != 0 ? ϕ((x-X[j])/h) : ψ((x-X[j])/h))
        dλ(x,j) = j ∈ exvals ? 0 : (j % 2 != 0 ? dϕ((x-X[j])/h)*1/h : dψ((x-X[j])/h)*1/h)
        #---- Assemble the matrix ----
        @info "Assembling the matrix"
        A = zeros(2n+1,2n+1)
        for i in tqdm(1:2n+1)
            for j in 1:2n+1
                A[i,j] = quadgk(x->dλ(x,i)*dλ(x,j),BigFloat(0),BigFloat(1))[1]
            end
        end
        display(A)
        #---- Assemble the vector ----
        @info "Assembling the vector"
        b = zeros(2n+1)
        for i in tqdm(1:2n+1)
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
        Uh(x) = sum(alpha[j]*λ(x,j) for j in 1:2n+1)
        Yh = Uh.(domain)
        #---- Calculate the error ----
        @info "Calculating the error"
        error = maximum(abs.(Yex-Yh))
        @info "Error: $error"
        #---- Save the results ----
        Results[n] = (domain,Yex,Yh,error)
        #=
        #plot δ functions
        plot()
        Xvals = range(0,stop=1,length=1000)
        for j in 1:2n+1
            plot!(Xvals,dλ.(Xvals,j))
        end
        gui()
        =#
    end
    
    # Plot the results in 4 subplots
    @info "Plotting the results"
    plot_array = []
    for n in N
        push!(plot_array,plot(Results[n][1],Results[n][2],label="Exact Solution",title="n=$n"))
        plot!(Results[n][1],Results[n][3],label="Numerical Solution")
    end
    p=plot(plot_array...,layout=(2,2),size=(800,800))
    gui()
    
end