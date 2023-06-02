import streamlit as st

def add_zeros(digits):
    current_rate = '.'.join(digits) + '000000'
    return current_rate

def main():
    st.title("Rate Calculator")
    digits = []
    for i in range(8):
        digit = st.text_input(f"Digit {i+1}", max_chars=1)
        digits.append(digit)

    if all(digits):
        result = add_zeros(digits)
        st.write(f"The current rate is: {result}")

if __name__ == '__main__':
    main()
