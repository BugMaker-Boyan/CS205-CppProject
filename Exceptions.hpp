//
// Created by Boyan on 2021/6/7.
//

#include <iostream>
#include <utility>
#include "windows.h"
#pragma once
using namespace std;

class BaseException {
public:
    virtual void Show() const {
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED);
        cout << "Unknown error." << endl;
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
    }
};

class MatrixShapeNotValidException : public BaseException{
private:
    int rows;
    int cols;
public:
    explicit MatrixShapeNotValidException(int rows, int cols){
        this->rows = rows;
        this->cols = cols;
    }

    void Show() const{
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED);
        cout<<"MatrixShapeNotValidException:\n"
            << "Both rows: "<< rows<<"\tcols: "<<cols<<" should be positive!"<<endl;
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
    }
};

class MatrixMultiplicationShapeNotValidException : public BaseException {
private:
    pair<int, int> matrix_1;
    pair<int, int> matrix_2;
public:
    explicit MatrixMultiplicationShapeNotValidException(pair<int, int> matrix_1, pair<int, int> matrix_2){
        this->matrix_1 = matrix_1;
        this->matrix_2 = matrix_2;
    }

    void Show() const{
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED);
        cout<<"MatrixMultiplicationShapeNotValidException:\n"
            << "Two matrix with shape "<<
            "("<<matrix_1.first<<", "<<matrix_1.second<<") and ("<<
            matrix_2.first<<", "<<matrix_2.second<<") can not do multiplication!"<<endl;
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
    }
};

class DefinedConjugationFunctionNotValidException : public BaseException {
private:
    string type_name;
public:
    explicit DefinedConjugationFunctionNotValidException(string type_name){
        this->type_name = move(type_name);
    }

    void Show(){
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED);
        cout<<"DefinedConjugationFunctionNotValidException:\n"
            << "The user-defined conjugation function for data type : "<<type_name<<" fail to work, "<<
            "the function should be defined like 'Type function_name(const Type& x)', please check further."<<endl;
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
    }
};

class DotMatrixShapeNotValid : public BaseException {
private:
    pair<int, int> matrix_1;
    pair<int, int> matrix_2;
public:
    explicit DotMatrixShapeNotValid(pair<int, int> matrix_1, pair<int, int> matrix_2){
        this->matrix_1 = matrix_1;
        this->matrix_2 = matrix_2;
    }

    void Show() const{
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED);
        cout<<"DotMatrixShapeNotValid:\n"
            << "Two matrix with shape "<<
            "("<<matrix_1.first<<", "<<matrix_1.second<<") and ("<<
            matrix_2.first<<", "<<matrix_2.second<<") can not do dot, they must be in the same shape."<<endl;
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
    }
};

class MatrixReshapeNotValidException : public BaseException {
private:
    int old_size;
    int new_size;
public:
    MatrixReshapeNotValidException(int old_size, int new_size){
        this->old_size = old_size;
        this->new_size = new_size;
    }

    void Show() const{
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED);
        cout<<"MatrixReshapeNotValidException:\n"
            << "The old matrix has "<<
            old_size<<" elements, "<<"but the new matrix has "<<
            new_size<<" elements, "<<"they must be the same."<<endl;
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
    }
};

class CrossProductVectorNotValidException : public BaseException {
private:
    pair<int, int> shape;
public:
    explicit CrossProductVectorNotValidException(pair<int, int> shape){
        this->shape = shape;
    }

    void Show() const{
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED);
        cout<<"CrossProductVectorNotValidException:\n"
            << "The cross-product need vector with shape (1, 3) or (3, 1), but input shape is "<<
            "("<<shape.first<<", "<<shape.second<<")."<<endl;
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
    }
};

class DeterminantMatrixNotSquareException : public BaseException {
private:
    int rows;
    int cols;
public:
    explicit DeterminantMatrixNotSquareException(int rows, int cols){
        this->rows = rows;
        this->cols = cols;
    }

    void Show() const{
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED);
        cout<<"DeterminantMatrixNotSquareException:\n"
            << "Rows: "<< rows<<"and cols: "<<cols<<" should be same number, since determinant needs a square matrix."<<endl;
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
    }
};

class MatrixInverseNotValidException : public BaseException {
private:
    string reason;
public:
    explicit MatrixInverseNotValidException(string reason){
        this->reason = move(reason);
    }

    void Show() const{
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED);
        cout<<"MatrixInverseNotValidException:\n"<<
        "The matrix has not inverse, since "<<reason<<" ."<<endl;
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
    }
};

class MatrixIndexNotValidException : public BaseException {
public:
    explicit MatrixIndexNotValidException() = default;

    void Show() const{
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED);
        cout<<"MatrixIndexNotValidException:\n"<<
            "The matrix with shape (m,n), row index in [1,m] and column index in [1,n]."<<endl;
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
    }
};

class MatrixTraceNotSquareException : public BaseException {
private:
    int rows;
    int cols;
public:
    explicit MatrixTraceNotSquareException(int rows, int cols){
        this->rows = rows;
        this->cols = cols;
    }

    void Show() const{
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED);
        cout<<"MatrixTraceNotSquareException:\n"
            << "Rows: "<< rows<<"and cols: "<<cols<<" should be same number, since trace needs a square matrix."<<endl;
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
    }
};

class MatrixShapeNotSameException : public BaseException {
public:
    explicit MatrixShapeNotSameException() = default;

    void Show() const{
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED);
        cout<<"MatrixShapeNotSameException:\n"
            << " Two matrix should have the same shape."<<endl;
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
    }
};

class MatrixConvolutionNotValidException : public BaseException {
public:
    explicit MatrixConvolutionNotValidException() = default;

    void Show() const {
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED);
        cout << "MatrixConvolutionNotValidException:\n"
            << "There exist an error during convolution." << endl;
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
    }
};

class MatrixEquationParameterException : public BaseException {
public:
    explicit MatrixEquationParameterException() = default;

    void Show() const {
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED);
        cout << "MatrixEquationParameterException:\n"
            << " A and b should have same number of rows and b should be an column vector." << endl;
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
    }
};

class MatrixEquationNoSolutionException : public BaseException {
public:
    explicit MatrixEquationNoSolutionException() = default;

    void Show() const {
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED);
        cout << "MatrixEquationNoSolutionException:\n"
            << " The equation has no solutions." << endl;
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
    }
};

class MatrixNotDenseException : public BaseException {
public:
    explicit MatrixNotDenseException() = default;

    void Show() const {
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED);
        cout << "MatrixNotDenseException:\n"
            << " The matrix to operate is not a dense matrix." << endl;
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
    }
};

class MatrixNotSparseException : public BaseException {
public:
    explicit MatrixNotSparseException() = default;

    void Show() const {
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED);
        cout << "MatrixNotSparseException:\n"
            << " The matrix to operate is not a sparse matrix." << endl;
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
    }
};