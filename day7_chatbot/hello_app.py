import streamlit as st

st.title("🎉 Hello, Streamlit!")
st.write("这是你的第一个 LLM 应用界面。")
user_input = st.text_input("请输入你的问题：")

if user_input:
    st.write(f"你输入了：{user_input}")
