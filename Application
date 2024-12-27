# pip install streamlit numpy pandas sumpy
import streamlit as st
import numpy as np
import pandas as pd

# 1. Set Page Config FIRST
st.set_page_config(
    page_title="Matrix Operations (Decimal Support)",
    layout="wide"
)

# 2. Minimal Global CSS (Optional)
CUSTOM_CSS = """
<style>
body, .stApp {
    background-color: #F7F8FA;
    color: black;
    font-family: 'Poppins', sans-serif;
}
/* Hide top menu and footer if you want */
#MainMenu, footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# 3. Matrix Operations
def add_matrices(matrix1, matrix2):
    try:
        np_matrix1 = np.array(matrix1, dtype=float)
        np_matrix2 = np.array(matrix2, dtype=float)
        return (np_matrix1 + np_matrix2).tolist()
    except Exception as e:
        return f"Error: {str(e)}"

def multiply_matrices(matrix1, matrix2):
    try:
        np_matrix1 = np.array(matrix1, dtype=float)
        np_matrix2 = np.array(matrix2, dtype=float)
        return np.dot(np_matrix1, np_matrix2).tolist()
    except Exception as e:
        return f"Error: {str(e)}"

def determinant(matrix):
    try:
        np_matrix = np.array(matrix, dtype=float)
        return round(np.linalg.det(np_matrix), 4)
    except Exception as e:
        return f"Error: {str(e)}"

def inverse(matrix):
    try:
        np_matrix = np.array(matrix, dtype=float)
        return np.linalg.inv(np_matrix).tolist()
    except Exception as e:
        return f"Error: {str(e)}"

def solve_linear_system(matrix, constants):
    try:
        np_matrix = np.array(matrix, dtype=float)
        np_constants = np.array(constants, dtype=float).flatten()
        solution = np.linalg.solve(np_matrix, np_constants)
        return solution.tolist()
    except Exception as e:
        return f"Error: {str(e)}"

def frobenius_norm(matrix):
    try:
        np_matrix = np.array(matrix, dtype=float)
        return round(np.linalg.norm(np_matrix, "fro"), 4)
    except Exception as e:
        return f"Error: {str(e)}"

def qr_decomposition(matrix):
    try:
        np_matrix = np.array(matrix, dtype=float)
        Q, R = np.linalg.qr(np_matrix)
        return Q.tolist(), R.tolist()
    except Exception as e:
        return f"Error: {str(e)}"

def trace(matrix):
    try:
        np_matrix = np.array(matrix, dtype=float)
        return np.trace(np_matrix)
    except Exception as e:
        return f"Error: {str(e)}"

def transpose(matrix):
    try:
        np_matrix = np.array(matrix, dtype=float)
        return np_matrix.T.tolist()
    except Exception as e:
        return f"Error: {str(e)}"

def matrix_rank(matrix):
    try:
        np_matrix = np.array(matrix, dtype=float)
        return np.linalg.matrix_rank(np_matrix)
    except Exception as e:
        return f"Error: {str(e)}"

def gauss_jordan_elimination(matrix):
    try:
        import sympy
        M = sympy.Matrix(matrix)
        rref_matrix, _ = M.rref()
        return rref_matrix.tolist()
    except Exception as e:
        return f"Error: {str(e)}"


# 4. Helpers to Add/Delete Rows/Columns
def add_row(matrix_key):
    arr = np.array(st.session_state[matrix_key], dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    new_row = np.zeros((1, arr.shape[1]))
    updated = np.vstack([arr, new_row])
    st.session_state[matrix_key] = updated.tolist()

def add_col(matrix_key):
    arr = np.array(st.session_state[matrix_key], dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    new_col = np.zeros((arr.shape[0], 1))
    updated = np.hstack([arr, new_col])
    st.session_state[matrix_key] = updated.tolist()

def remove_row(matrix_key):
    arr = np.array(st.session_state[matrix_key], dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[0] > 1:
        updated = arr[:-1, :]
        st.session_state[matrix_key] = updated.tolist()

def remove_col(matrix_key):
    arr = np.array(st.session_state[matrix_key], dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] > 1:
        updated = arr[:, :-1]
        st.session_state[matrix_key] = updated.tolist()


# 5. Initialize Session State
if "matrix1" not in st.session_state:
    st.session_state["matrix1"] = [[1, 2], [3, 4]]
if "matrix2" not in st.session_state:
    st.session_state["matrix2"] = [[5, 6], [7, 8]]
if "constants" not in st.session_state:
    st.session_state["constants"] = [[1], [1]]
if "result" not in st.session_state:
    st.session_state["result"] = []
if "scalar_result" not in st.session_state:
    st.session_state["scalar_result"] = ""


# 6. Helper: Build Column Config for Decimals
def _get_decimal_column_config(df: pd.DataFrame):
    """
    Returns a dict that configures each column as a NumberColumn
    with a small step (0.01), allowing decimal input.
    """
    col_config = {}
    for col_name in df.columns:
        col_config[col_name] = st.column_config.NumberColumn(
            label=f"{col_name}",  # or a nicer label
            step=0.01,            # step size for decimals
            format="%.4f",        # show up to 4 decimal places (optional)
        )
    return col_config


# 7. Main App
def main():
    st.title("Matrix Operations")

    # ---- Sidebar (Radio Buttons Menu) ----
    st.sidebar.title("Menu")
    operations = [
        "Addition",
        "Multiplication",
        "Determinant",
        "Inverse",
        "Solve Linear System",
        "Frobenius Norm",
        "QR Decomposition",
        "Trace",
        "Transpose",
        "Rank",
        "Gauss Jordan Elimination",
    ]
    operation = st.sidebar.radio("Select an Operation:", operations)

    # Layout for matrices
    col1, col2, col3 = st.columns([1.1, 1.1, 1])

    # --- Matrix 1 ---
    with col1:
        st.subheader("Matrix 1")
        # Convert session_state "matrix1" to a DataFrame
        df_m1 = pd.DataFrame(st.session_state["matrix1"])

        # Build decimal column config
        decimal_config_m1 = _get_decimal_column_config(df_m1)

        # Show the data editor with decimal config
        df_m1_edited = st.data_editor(
            df_m1,
            key="matrix1_editor",
            column_config=decimal_config_m1,
            use_container_width=True
        )
        # After user edits, store back to session_state
        st.session_state["matrix1"] = df_m1_edited.to_numpy().tolist()

        # Buttons for add/del row/col
        c1_1, c1_2, c1_3, c1_4 = st.columns(4)
        with c1_1:
            if st.button("➕Row (M1)"):
                add_row("matrix1")
        with c1_2:
            if st.button("➕Col (M1)"):
                add_col("matrix1")
        with c1_3:
            if st.button("➖Row (M1)"):
                remove_row("matrix1")
        with c1_4:
            if st.button("➖Col (M1)"):
                remove_col("matrix1")

    # --- Matrix 2 ---
    with col2:
        st.subheader("Matrix 2")
        if operation in ["Addition", "Multiplication"]:
            df_m2 = pd.DataFrame(st.session_state["matrix2"])
            decimal_config_m2 = _get_decimal_column_config(df_m2)

            df_m2_edited = st.data_editor(
                df_m2,
                key="matrix2_editor",
                column_config=decimal_config_m2,
                use_container_width=True
            )
            st.session_state["matrix2"] = df_m2_edited.to_numpy().tolist()

            d1, d2, d3, d4 = st.columns(4)
            with d1:
                if st.button("➕Row (M2)"):
                    add_row("matrix2")
            with d2:
                if st.button("➕Col (M2)"):
                    add_col("matrix2")
            with d3:
                if st.button("➖Row (M2)"):
                    remove_row("matrix2")
            with d4:
                if st.button("➖Col (M2)"):
                    remove_col("matrix2")
        else:
            st.info("Matrix 2 not needed for this operation.")

    # --- Constants ---
    with col3:
        st.subheader("Constants")
        if operation == "Solve Linear System":
            df_const = pd.DataFrame(st.session_state["constants"])
            decimal_config_const = _get_decimal_column_config(df_const)

            df_const_edited = st.data_editor(
                df_const,
                key="constants_editor",
                column_config=decimal_config_const,
                use_container_width=True
            )
            st.session_state["constants"] = df_const_edited.to_numpy().tolist()

            e1, e2, e3, e4 = st.columns(4)
            with e1:
                if st.button("➕Row (Const)"):
                    add_row("constants")
            with e2:
                if st.button("➕Col (Const)"):
                    add_col("constants")
            with e3:
                if st.button("➖Row (Const)"):
                    remove_row("constants")
            with e4:
                if st.button("➖Col (Const)"):
                    remove_col("constants")
        else:
            st.info("Constants not needed for this operation.")

    st.divider()

    # --- Compute Button ---
    if st.button("Compute"):
        mat1 = st.session_state["matrix1"]
        mat2 = st.session_state["matrix2"]
        const = st.session_state["constants"]

        if operation == "Addition":
            st.session_state["result"] = add_matrices(mat1, mat2)
            st.session_state["scalar_result"] = ""
        elif operation == "Multiplication":
            st.session_state["result"] = multiply_matrices(mat1, mat2)
            st.session_state["scalar_result"] = ""
        elif operation == "Determinant":
            st.session_state["scalar_result"] = determinant(mat1)
            st.session_state["result"] = []
        elif operation == "Inverse":
            st.session_state["result"] = inverse(mat1)
            st.session_state["scalar_result"] = ""
        elif operation == "Solve Linear System":
            st.session_state["scalar_result"] = solve_linear_system(mat1, const)
            st.session_state["result"] = []
        elif operation == "Frobenius Norm":
            st.session_state["scalar_result"] = frobenius_norm(mat1)
            st.session_state["result"] = []
        elif operation == "QR Decomposition":
            Q, R = qr_decomposition(mat1)
            st.session_state["result"] = Q
            st.session_state["scalar_result"] = R
        elif operation == "Trace":
            st.session_state["scalar_result"] = trace(mat1)
            st.session_state["result"] = []
        elif operation == "Transpose":
            st.session_state["result"] = transpose(mat1)
            st.session_state["scalar_result"] = ""
        elif operation == "Rank":
            st.session_state["scalar_result"] = matrix_rank(mat1)
            st.session_state["result"] = []
        elif operation == "Gauss Jordan Elimination":
            st.session_state["result"] = gauss_jordan_elimination(mat1)
            st.session_state["scalar_result"] = ""

    # --- Display Results ---
    st.subheader("Results")

    # If we have a matrix result
    if isinstance(st.session_state["result"], list) and len(st.session_state["result"]) > 0:
        df_result = pd.DataFrame(st.session_state["result"])
        # Basic styling for bigger text
        df_styled = df_result.style.set_properties(
            **{
                'font-size': '18px',
                'color': 'black',
                'font-family': 'Poppins, sans-serif'
            }
        )
        st.dataframe(df_styled, use_container_width=True)

    # If we have a scalar/vector result
    if st.session_state["scalar_result"] != "":
        # Convert to string if list
        if isinstance(st.session_state["scalar_result"], list):
            scalar_str = str(st.session_state["scalar_result"])
        else:
            scalar_str = str(st.session_state["scalar_result"])

        st.markdown(
            f"""
            <div style="color: black; font-size: 18px; font-family: 'Poppins', sans-serif;">
                {scalar_str}
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
