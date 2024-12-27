import streamlit as st
import numpy as np
import pandas as pd

# ===============================
# 1. SET PAGE CONFIG FIRST
# ===============================
st.set_page_config(
    page_title="Matrix Operations – Immediate Update Fix",
    layout="wide"
)

# ===============================
# 2. GLOBAL CSS (Optional)
# ===============================
CUSTOM_CSS = """
<style>
body, .stApp {
    background-color: #F7F8FA;
    color: black;
    font-family: 'Poppins', sans-serif;
}
#MainMenu, footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ===============================
# 3. MATRIX OPERATIONS
# ===============================


def add_matrices(matrix1, matrix2):
    try:
        a = np.array(matrix1, dtype=float)
        b = np.array(matrix2, dtype=float)
        return (a + b).tolist()
    except Exception as e:
        return f"Error: {str(e)}"


def multiply_matrices(matrix1, matrix2):
    try:
        a = np.array(matrix1, dtype=float)
        b = np.array(matrix2, dtype=float)
        return np.dot(a, b).tolist()
    except Exception as e:
        return f"Error: {str(e)}"


def determinant(matrix):
    try:
        a = np.array(matrix, dtype=float)
        return round(np.linalg.det(a), 4)
    except Exception as e:
        return f"Error: {str(e)}"


def inverse(matrix):
    try:
        a = np.array(matrix, dtype=float)
        return np.linalg.inv(a).tolist()
    except Exception as e:
        return f"Error: {str(e)}"


def solve_linear_system(matrix, constants):
    try:
        a = np.array(matrix, dtype=float)
        c = np.array(constants, dtype=float).flatten()
        sol = np.linalg.solve(a, c)
        return sol.tolist()
    except Exception as e:
        return f"Error: {str(e)}"


def frobenius_norm(matrix):
    try:
        a = np.array(matrix, dtype=float)
        return round(np.linalg.norm(a, "fro"), 4)
    except Exception as e:
        return f"Error: {str(e)}"


def qr_decomposition(matrix):
    try:
        a = np.array(matrix, dtype=float)
        Q, R = np.linalg.qr(a)
        return Q.tolist(), R.tolist()
    except Exception as e:
        return f"Error: {str(e)}"


def trace(matrix):
    try:
        a = np.array(matrix, dtype=float)
        return np.trace(a)
    except Exception as e:
        return f"Error: {str(e)}"


def transpose(matrix):
    try:
        a = np.array(matrix, dtype=float)
        return a.T.tolist()
    except Exception as e:
        return f"Error: {str(e)}"


def matrix_rank(matrix):
    try:
        a = np.array(matrix, dtype=float)
        return np.linalg.matrix_rank(a)
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

# ===============================
# 4. HELPERS TO ADD/DELETE ROWS/COLS
#    We call st.experimental_rerun() to refresh instantly.
# ===============================


def add_row(matrix_key):
    df = st.session_state[matrix_key]
    new_row = pd.DataFrame([[0.0]*df.shape[1]], columns=df.columns)
    st.session_state[matrix_key] = pd.concat([df, new_row], ignore_index=True)
    # Also update list-of-lists version
    _update_list_matrix(matrix_key)
    st.experimental_rerun()


def add_col(matrix_key):
    df = st.session_state[matrix_key]
    new_col_name = f"col{df.shape[1]}"  # or any naming scheme
    df[new_col_name] = 0.0
    st.session_state[matrix_key] = df
    _update_list_matrix(matrix_key)
    st.experimental_rerun()


def remove_row(matrix_key):
    df = st.session_state[matrix_key]
    if df.shape[0] > 1:
        df = df.iloc[:-1]
        st.session_state[matrix_key] = df
        _update_list_matrix(matrix_key)
    st.experimental_rerun()


def remove_col(matrix_key):
    df = st.session_state[matrix_key]
    if df.shape[1] > 1:
        df = df.iloc[:, :-1]
        st.session_state[matrix_key] = df
        _update_list_matrix(matrix_key)
    st.experimental_rerun()


def _update_list_matrix(matrix_key):
    """
    Convert the current DataFrame in session_state[matrix_key]
    to a list-of-lists and store it in session_state[matrix_key+'_list'].
    """
    df = st.session_state[matrix_key]
    st.session_state[matrix_key + "_list"] = df.to_numpy().tolist()

# ===============================
# 5. INITIALIZE SESSION STATE
#    We keep both a DataFrame (for st.data_editor) and a list-of-lists.
# ===============================


def init_session_state_matrix(df_key, list_key, initial_data):
    """
    If the DataFrame key or the list-of-lists key doesn't exist yet,
    create them from the given initial_data.
    """
    if df_key not in st.session_state:
        st.session_state[df_key] = pd.DataFrame(initial_data)
    if list_key not in st.session_state:
        st.session_state[list_key] = initial_data


if "result" not in st.session_state:
    st.session_state["result"] = []
if "scalar_result" not in st.session_state:
    st.session_state["scalar_result"] = ""

# Matrix 1
init_session_state_matrix(
    df_key="matrix1",
    list_key="matrix1_list",
    initial_data=[[1, 2], [3, 4]]
)
# Matrix 2
init_session_state_matrix(
    df_key="matrix2",
    list_key="matrix2_list",
    initial_data=[[5, 6], [7, 8]]
)
# Constants
init_session_state_matrix(
    df_key="constants",
    list_key="constants_list",
    initial_data=[[1], [1]]
)

# ===============================
# 6. COLUMN CONFIG FOR DECIMALS (optional)
# ===============================


def get_decimal_config(df):
    col_config = {}
    for col in df.columns:
        col_config[col] = st.column_config.NumberColumn(
            label=str(col),
            step=0.01,
            format="%.4f",
        )
    return col_config

# ===============================
# 7. MAIN APP
# ===============================


def main():
    st.title("Immediate Update Fix: Matrix Operations")

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

    col1, col2, col3 = st.columns([1.1, 1.1, 1])

    # --------------------------------
    # Matrix 1
    # --------------------------------
    with col1:
        st.subheader("Matrix 1")

        df_m1 = st.session_state["matrix1"]
        config_m1 = get_decimal_config(df_m1)

        edited_df_m1 = st.data_editor(
            df_m1,
            column_config=config_m1,
            key="matrix1_editor",
            use_container_width=True
        )
        # Update the DF in session_state if user changed anything
        if not edited_df_m1.equals(df_m1):
            st.session_state["matrix1"] = edited_df_m1
            _update_list_matrix("matrix1")

        # Buttons
        b1_1, b1_2, b1_3, b1_4 = st.columns(4)
        with b1_1:
            if st.button("➕Row (M1)"):
                add_row("matrix1")
        with b1_2:
            if st.button("➕Col (M1)"):
                add_col("matrix1")
        with b1_3:
            if st.button("➖Row (M1)"):
                remove_row("matrix1")
        with b1_4:
            if st.button("➖Col (M1)"):
                remove_col("matrix1")

    # --------------------------------
    # Matrix 2
    # --------------------------------
    with col2:
        st.subheader("Matrix 2")
        if operation in ["Addition", "Multiplication"]:
            df_m2 = st.session_state["matrix2"]
            config_m2 = get_decimal_config(df_m2)

            edited_df_m2 = st.data_editor(
                df_m2,
                column_config=config_m2,
                key="matrix2_editor",
                use_container_width=True
            )
            # Update session_state if changed
            if not edited_df_m2.equals(df_m2):
                st.session_state["matrix2"] = edited_df_m2
                _update_list_matrix("matrix2")

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

    # --------------------------------
    # Constants
    # --------------------------------
    with col3:
        st.subheader("Constants")
        if operation == "Solve Linear System":
            df_const = st.session_state["constants"]
            config_const = get_decimal_config(df_const)

            edited_df_const = st.data_editor(
                df_const,
                column_config=config_const,
                key="constants_editor",
                use_container_width=True
            )
            if not edited_df_const.equals(df_const):
                st.session_state["constants"] = edited_df_const
                _update_list_matrix("constants")

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

    # --------------------------------
    # Compute the selected operation
    # --------------------------------
    if st.button("Compute"):
        mat1 = st.session_state["matrix1_list"]
        mat2 = st.session_state["matrix2_list"]
        const = st.session_state["constants_list"]

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
            st.session_state["scalar_result"] = solve_linear_system(
                mat1, const)
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

    # --------------------------------
    # Display Results
    # --------------------------------
    st.subheader("Results")
    if isinstance(st.session_state["result"], list) and len(st.session_state["result"]) > 0:
        df_result = pd.DataFrame(st.session_state["result"])
        df_styled = df_result.style.set_properties(
            **{'font-size': '18px', 'color': 'black', 'font-family': 'Poppins, sans-serif'}
        )
        st.dataframe(df_styled, use_container_width=True)

    if st.session_state["scalar_result"] != "":
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
