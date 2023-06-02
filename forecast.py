import streamlit as st

def add_zeros(current_rate):
    return f'{current_rate:.6f}'

def main():
    st.title("Rate Calculator")
    current_rate = st.text_input("Enter the rate (e.g., 12.345678):")
    if current_rate:
        result = add_zeros(float(current_rate))
        st.write(f"The current rate is: {result}")

if __name__ == '__main__':
    main()
