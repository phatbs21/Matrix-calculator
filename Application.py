import streamlit as st
import numpy as np
import pandas as pd

# -----------------------------
# 1) PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Matrix calculation",
    layout="wide"
)

# -----------------------------
# 2) GLOBAL CSS (Optional)
# -----------------------------
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


# -----------------------------
# 3) MATRIX OPERATIONS
# -----------------------------
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


# -----------------------------
# 4) INITIALIZE SESSION STATE
# -----------------------------
def init_df_state(df_key, list_key, data):
    if df_key not in st.session_state:
        st.session_state[df_key] = pd.DataFrame(data)
    if list_key not in st.session_state:
        st.session_state[list_key] = data


init_df_state("matrix1_df", "matrix1_list", [[1, 2], [3, 4]])
init_df_state("matrix2_df", "matrix2_list", [[5, 6], [7, 8]])
init_df_state("constants_df", "constants_list", [[1], [1]])

if "result" not in st.session_state:
    st.session_state["result"] = []
if "scalar_result" not in st.session_state:
    st.session_state["scalar_result"] = ""


# -----------------------------
# 5) ADD/DEL ROW/COL HELPER
#    (forces immediate rerun)
# -----------------------------
def _update_list_matrix(df_key, list_key):
    st.session_state[list_key] = st.session_state[df_key].to_numpy().tolist()


def add_row(df_key, list_key):
    df = st.session_state[df_key]
    new_row = pd.DataFrame([[0.0]*df.shape[1]], columns=df.columns)
    st.session_state[df_key] = pd.concat([df, new_row], ignore_index=True)
    _update_list_matrix(df_key, list_key)
    st.rerun()


def add_col(df_key, list_key):
    df = st.session_state[df_key]
    new_col_name = f"C{df.shape[1]}"
    df[new_col_name] = 0.0
    st.session_state[df_key] = df
    _update_list_matrix(df_key, list_key)
    st.rerun()


def remove_row(df_key, list_key):
    df = st.session_state[df_key]
    if df.shape[0] > 1:
        st.session_state[df_key] = df.iloc[:-1, :]
        _update_list_matrix(df_key, list_key)
    st.rerun()


def remove_col(df_key, list_key):
    df = st.session_state[df_key]
    if df.shape[1] > 1:
        st.session_state[df_key] = df.iloc[:, :-1]
        _update_list_matrix(df_key, list_key)
    st.rerun()


# -----------------------------
# 6) OPTIONAL: COLUMN CONFIG FOR DECIMALS
# -----------------------------
def get_decimal_config(df: pd.DataFrame):
    col_config = {}
    for c in df.columns:
        col_config[c] = st.column_config.NumberColumn(
            label=str(c),
            step=0.01,
            format="%.4f",
        )
    return col_config


# -----------------------------
# 7) MAIN APP
# -----------------------------
def main():
    st.title("Matrix calculation")

    # ~~~~~~~~~~~ Sidebar Menu ~~~~~~~~~~~
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

    # 7.1 Layout for the 3 data editors
    c1, c2, c3 = st.columns([1.1, 1.1, 1])

    # ----------- Matrix 1 -----------
    with c1:
        st.subheader("Matrix 1")
        df_m1 = st.session_state["matrix1_df"]

        # Show data editor
        config_m1 = get_decimal_config(df_m1)
        edited_m1 = st.data_editor(
            df_m1,
            key="m1_editor",
            column_config=config_m1,
            use_container_width=True,
        )
        # Show a "Submit Edits" button that re-runs, storing changes
        if st.button("Submit Edits (M1)"):
            st.session_state["matrix1_df"] = edited_m1
            _update_list_matrix("matrix1_df", "matrix1_list")
            st.rerun()

        # Row/Col Buttons
        b1_1, b1_2, b1_3, b1_4 = st.columns(4)
        with b1_1:
            if st.button("➕Row (M1)"):
                add_row("matrix1_df", "matrix1_list")
        with b1_2:
            if st.button("➕Col (M1)"):
                add_col("matrix1_df", "matrix1_list")
        with b1_3:
            if st.button("➖Row (M1)"):
                remove_row("matrix1_df", "matrix1_list")
        with b1_4:
            if st.button("➖Col (M1)"):
                remove_col("matrix1_df", "matrix1_list")

    # ----------- Matrix 2 -----------
    with c2:
        st.subheader("Matrix 2")
        if operation in ["Addition", "Multiplication"]:
            df_m2 = st.session_state["matrix2_df"]
            config_m2 = get_decimal_config(df_m2)
            edited_m2 = st.data_editor(
                df_m2,
                key="m2_editor",
                column_config=config_m2,
                use_container_width=True
            )
            if st.button("Submit Edits (M2)"):
                st.session_state["matrix2_df"] = edited_m2
                _update_list_matrix("matrix2_df", "matrix2_list")
                st.rerun()

            d2_1, d2_2, d2_3, d2_4 = st.columns(4)
            with d2_1:
                if st.button("➕Row (M2)"):
                    add_row("matrix2_df", "matrix2_list")
            with d2_2:
                if st.button("➕Col (M2)"):
                    add_col("matrix2_df", "matrix2_list")
            with d2_3:
                if st.button("➖Row (M2)"):
                    remove_row("matrix2_df", "matrix2_list")
            with d2_4:
                if st.button("➖Col (M2)"):
                    remove_col("matrix2_df", "matrix2_list")

        else:
            st.info("Matrix 2 not needed for this operation.")

    # ----------- Constants -----------
    with c3:
        st.subheader("Constants")
        if operation == "Solve Linear System":
            df_const = st.session_state["constants_df"]
            config_const = get_decimal_config(df_const)
            edited_const = st.data_editor(
                df_const,
                key="const_editor",
                column_config=config_const,
                use_container_width=True
            )
            if st.button("Submit Edits (Const)"):
                st.session_state["constants_df"] = edited_const
                _update_list_matrix("constants_df", "constants_list")
                st.rerun()

            e1, e2, e3, e4 = st.columns(4)
            with e1:
                if st.button("➕Row (Const)"):
                    add_row("constants_df", "constants_list")
            with e2:
                if st.button("➕Col (Const)"):
                    add_col("constants_df", "constants_list")
            with e3:
                if st.button("➖Row (Const)"):
                    remove_row("constants_df", "constants_list")
            with e4:
                if st.button("➖Col (Const)"):
                    remove_col("constants_df", "constants_list")
        else:
            st.info("Constants not needed for this operation.")

    st.divider()

    # 7.2: Compute Operation
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

    # 7.3: Display Results
    st.subheader("Results")
    if isinstance(st.session_state["result"], list) and len(st.session_state["result"]) > 0:
        df_res = pd.DataFrame(st.session_state["result"])
        df_styled = df_res.style.set_properties(
            **{'font-size': '18px', 'color': 'black', 'font-family': 'Poppins, sans-serif'}
        )
        st.dataframe(df_styled, use_container_width=True)

    if st.session_state["scalar_result"] != "":
        text = st.session_state["scalar_result"]
        if isinstance(text, list):
            text = str(text)
        st.markdown(
            f"""
            <div style='font-size:18px; color:black; font-family:Poppins,sans-serif;'>
            {text}
            </div>
            """,
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
