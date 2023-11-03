## 目录

- [简介](#简介)
- [使用说明](#使用说明)
- [注意事项](#注意事项)

## 简介

这个项目生成 $10\times 10$ 随机矩阵和密度为 $0.001$ 的 $10000\times 10000$ 稀疏矩阵，然后分别使用库函数、幂法、QR法、Arnoldi算法对其进行特征值的求解。这个项目有两个版本，一个是对称阵版本，另一个是非对称阵版本，分别有一些优劣，最后我们的项目会以UI界面的形式输出结果。

## 使用说明

- 确保你已经安装了 `requirement.txt` 中所列出的 Python 库。  
   
- 运行同级目录下的 `sparse_matrix_sym.py` 或者 `sprase_matrix_asym.py` 文件即可实现各题的解答。其中 `sparse_matrix_sym.py` 是实对称阵版本，避免了复数运算，计算更快，但为了减少运算量我在生成对称的 $10000\times 10000$ 随机稀疏矩阵时使用了一点小技巧，但可能会导致密度不是精准的 $0.001$，会有一点点小浮动。而 `sprase_matrix_asym.py` 是非对称阵版本，特征值的计算会出现大量复数，导致计算稍慢，而且我的函数目前还无法进行准确的复数计算，可能导致复数特征值计算结果不准确。
  
- 我现在将题目3的第(2)题计算部分的代码注释掉了，因为使用 QR法 解 $10000\times 10000$ 的稀疏矩阵特征值会导致运行很慢，强烈建议先把 `matrix_2` 改为一个更小的矩阵再取消我注释掉的两行代码，你可以在 `run_code` 函数中找到它们。

- 目前我采用 UI 来输出结果，如果你想在终端输出结果，可以删去或注释 UI 部分代码，然后删去所有的三引号 `"""` ，这样结果就会输出在终端。

- 更多信息请参考我的大作业报告pdf
## 注意事项

- 建议在 Google 商店中下载 MathJax 3 Plugin for Github 或类似的插件以便你的 Github 的 markdown 编辑器可以渲染 Latex 数学公式。
- 运行环境为 Python 3.11.5。
- 请确保你已经安装了 `requirement.txt` 中所列出的 Python 库。
- 建议在 VSCode 中运行代码，以避免中文乱码等问题。
- 请在充电状态下运行或开启性能模式，以避免运行速度过慢。

---

## Table of Contents

- [Introduction](#introduction)
- [Usage Instructions](#usage-instructions)
- [Notes](#notes)

## Introduction

This project generates a $10\times 10$ random matrix and a sparse matrix of size $10000\times 10000$ with a density of $0.001$. It then uses library functions, power method, QR method, and Arnoldi algorithm to solve for their eigenvalues. There are two versions of this project, one for symmetric matrices and the other for non-symmetric matrices, each with its own advantages and disadvantages. The project will output the results in the form of a UI interface.

## Usage Instructions

- Make sure you have installed the Python libraries listed in `requirement.txt`.
   
- Run the `sparse_matrix_sym.py` or `sprase_matrix_asym.py` file in the same directory to solve the problems. `sparse_matrix_sym.py` is the version for symmetric matrices, which avoids complex calculations and is faster, but to reduce the amount of computation, I used a small trick when generating the symmetric $10000\times 10000$ random sparse matrix, which may cause the density to be not exactly $0.001$, with a small fluctuation. `sprase_matrix_asym.py` is the version for non-symmetric matrices, where the calculation of eigenvalues will involve a large number of complex numbers, resulting in slightly slower computation, and my function currently cannot perform accurate complex number calculations, which may cause inaccurate results for complex eigenvalue calculations.
  
- I have commented out the code for the calculation part of problem 3, question (2), because using the QR method to solve for the eigenvalues of a sparse matrix of size $10000\times 10000$ will cause the program to run very slowly. I strongly recommend changing `matrix_2` to a smaller matrix before uncommenting the two lines of code I commented out, which you can find in the `run_code` function.

- Currently, I use a UI to output the results. If you want to output the results in the terminal, you can delete or comment out the UI part of the code, and then delete all the triple quotes `"""`, so that the results will be output in the terminal.

- For more information, please refer to my report in pdf format.

## Notes

- It is recommended to download the MathJax 3 Plugin for Github or a similar plugin from the Google Store so that your Github markdown editor can render Latex mathematical formulas.
- The operating environment is Python 3.11.5.
- Make sure you have installed the Python libraries listed in `requirement.txt`.
- It is recommended to run the code in VSCode to avoid problems such as Chinese garbled characters.
- Please run the program while charging or in performance mode to avoid slow running speed.
