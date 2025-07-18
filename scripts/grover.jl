using LinearAlgebra
using Yao
using Yao.EasyBuild: variational_circuit

function grover_step!(reg::AbstractRegister, oracle, U::AbstractBlock)
    apply!(reg |> oracle, reflect_circuit(U))
end

function reflect_circuit(gen::AbstractBlock)
    N = nqubits(gen)
    reflect0 = control(N, -collect(1:N-1), N=>-Z)
    chain(gen', reflect0, gen)
end

function solution_state(oracle, gen::AbstractBlock)
    N = nqubits(gen)
    reg = zero_state(N) |> gen
    reg.state[real.(statevec(ArrayReg(ones(ComplexF64, 1<<N)) |> oracle)) .> 0] .= 0
    normalize!(reg)
end

function num_grover_step(oracle, gen::AbstractBlock)
    N = nqubits(gen)
    reg = zero_state(N) |> gen
    ratio = abs2(solution_state(oracle, gen)' * reg)
    Int(round(pi/4/sqrt(ratio))) - 1
end
