using DrWatson
@quickactivate "TQK"

using Pkg
using PyCall
using Conda

function setup_python_environment()
    println("Setting up Python environment for Qiskit...")
    
    # Check if we need to use Conda
    if PyCall.conda
        println("Using Conda Python environment")
        
        # Install required packages
        Conda.add("pip")
        Conda.pip_interop(true)
        Conda.pip("install", "qiskit")
        Conda.pip("install", "qiskit-machine-learning")
        Conda.pip("install", "qiskit-aer")  # For simulation
    else
        # Using system Python
        println("Using system Python at: $(PyCall.python)")
        println("Make sure qiskit and qiskit-machine-learning are installed!")
        
        # Try to install via pip
        run(`$(PyCall.python) -m pip install qiskit qiskit-machine-learning qiskit-aer`)
    end
    
    # Verify installation
    try
        py"""
        import qiskit
        import qiskit_machine_learning
        print(f"Qiskit version: {qiskit.__version__}")
        print(f"Qiskit ML version: {qiskit_machine_learning.__version__}")
        """
        println("✓ Python environment setup complete!")
    catch e
        error("Failed to import required packages: $e")
    end
end

# Run setup
setup_python_environment()
