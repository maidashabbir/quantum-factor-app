import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
from io import BytesIO

# Setup

st.set_page_config(page_title="Quantum vs Classical Factorization", layout="centered")
st.title("🔐 Quantum vs Classical Factorization Simulator")
st.markdown("""
This app demonstrates how quantum computing (via Shor's Algorithm) compares to classical trial division when factoring composite numbers.
- 🧮 Classical: Works on any number
- ⚛️ Quantum: Simulates Shor's Algorithm (only for N=15)
""")


# Classical Factorization

def trial_division(n):
    start = time.time()
    factors = []
    for i in range(2, n):
        while n % i == 0:
            factors.append(i)
            n //= i
    end = time.time()
    return factors, round(end - start, 6)


# Quantum Functions (Shor's Algorithm for N=15)

def c_amod15(a, power):
    U = QuantumCircuit(4)
    for _ in range(power):
        if a in [2, 13]:
            U.swap(0, 1)
            U.swap(1, 2)
            U.swap(2, 3)
        elif a in [7, 8]:
            U.swap(2, 3)
            U.swap(1, 2)
            U.swap(0, 1)
        elif a == 11:
            U.swap(1, 2)
            U.swap(0, 1)
            U.swap(2, 3)
        elif a == 4:
            U.swap(0, 2)
            U.swap(1, 3)
    return U.to_gate().control()

def qft_dagger(n):
    qc = QuantumCircuit(n)
    for q in range(n // 2):
        qc.swap(q, n - q - 1)
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi / float(2 ** (j - m)), m, j)
        qc.h(j)
    return qc

def qpe_amod15(a):
    n_count = 8
    qc = QuantumCircuit(4 + n_count, n_count)
    for q in range(n_count):
        qc.h(q)
    qc.x(n_count + 1)
    for q in range(n_count):
        qc.append(c_amod15(a, 2**q), [q] + [i + n_count for i in range(4)])
    qc.append(qft_dagger(n_count), range(n_count))
    qc.measure(range(n_count), range(n_count))

    backend = Aer.get_backend('aer_simulator')
    transpiled = transpile(qc, backend=backend)
    result = backend.run(transpiled, shots=1024).result()
    return result.get_counts()


# Sidebar Interaction

number = st.sidebar.number_input("Select a number to factor (15 is quantum-enabled):", min_value=10, max_value=100, value=15)
run_quantum = st.sidebar.checkbox("⚛️ Run Quantum Simulation (only for N=15)", value=True if number == 15 else False)


# Run Classical Factorization

classical_factors, classical_time = trial_division(number)
st.subheader("🧮 Classical Result")
st.write(f"**Factors:** {classical_factors}")
st.write(f"**Time:** {classical_time} seconds")


# Run Quantum Simulation (Only for N=15)

if run_quantum and number == 15:
    st.subheader("⚛️ Quantum (Shor's Algorithm) Result")
    start = time.time()
    counts = qpe_amod15(7)
    quantum_time = round(time.time() - start, 6)
    st.write("**Expected Factors:** [3, 5]")
    st.write(f"**Time:** {quantum_time} seconds")

    # Show histogram
    st.pyplot(plot_histogram(counts, title="Quantum Measurement Results"))
else:
    st.subheader("⚛️ Quantum Simulation")
    st.info("Shor's algorithm simulation is only available for N = 15.")


# Batch Comparison Chart (Classical)

with st.expander("📊 Compare Classical Performance on Multiple Numbers"):
    test_numbers = [15, 21, 33, 35, 39, 51, 57, 65, 77, 85, 91, 95, 99]
    results = []
    for n in test_numbers:
        f, t = trial_division(n)
        results.append({'Number': n, 'Factors': f, 'Time': t})

    df = pd.DataFrame(results)
    st.dataframe(df)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df['Number'], df['Time'], color='skyblue')
    ax.set_title("Classical Factorization Time Comparison")
    ax.set_xlabel("Number")
    ax.set_ylabel("Time (seconds)")
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

    # Download CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("💾 Download Results as CSV", data=csv, file_name="factorization_results.csv", mime='text/csv')


# Reflection / Footer

st.markdown("""
---
### 💬 Reflection
This interactive artefact demonstrates how quantum computers could break RSA encryption by factoring integers exponentially faster than classical methods. While current quantum tech is limited (e.g., only N=15 here), the simulation showcases the potential of Shor’s Algorithm in a post-quantum future.

> 🔐 **Did You Know?** RSA-2048 would take classical computers millions of years to crack — but a quantum computer could do it in minutes (if scaled).
""")