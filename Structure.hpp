#pragma once
#include <vector>
#include <iostream>
#include "Exceptions.hpp"
#include <complex>
#include <algorithm>
#include <functional>
#include <numeric>
#include <set>
#include <opencv2/opencv.hpp>
#include <stack>
#include <map>
using namespace std;
using namespace cv;

template <typename Type>
class AbstractTensor{

};

template <typename Type>
class Matrix: public AbstractTensor<Type>{
public:
    // default constructor
    explicit Matrix():Matrix(1,1){};

    // constructor with rows and cols
    explicit Matrix(int rows, int cols);

    // constructor with rows, cols and initial val
    explicit Matrix(int rows, int cols, Type val);

    // constructor with const vector
    explicit Matrix(const vector<vector<Type>>& vec);

    // constructor with const map
    explicit Matrix(map<pair<int, int>, Type>& data_sparse);

    // copy constructor
    Matrix(const Matrix<Type>& matrix);

    // assignment operator
    Matrix<Type>& operator=(const Matrix<Type>& matrix);

    // destructor
    ~Matrix();

    static Matrix<Type> zeros(int rows, int cols);

    static Matrix<Type> ones(int rows, int cols);

    static Matrix<Type> values(int rows, int cols, Type val);

    static Matrix<Type> eyes(int n, int val=1);

    pair<int, int> shape() const;

    int size() const;

    int cols() const;

    int rows() const;

    bool is_square() const;

    bool is_vector() const;

    Matrix<Type> reshape(int rows, int cols, bool inplace=false);

    Matrix<Type> operator+(const Matrix<Type>& matrix);

    Matrix<Type> operator+(const Type& val);

    Matrix<Type> operator-(const Matrix<Type>& matrix);

    Matrix<Type> operator-(const Type& val);

    Matrix<Type> operator*(const Matrix<Type>& matrix);

    Matrix<Type> operator*(const Type& val);

    Matrix<Type> operator/(const Type& val);

    Matrix<Type> transposition(bool inplace=false);

    Matrix<Type> conjugation(bool inplace=false);

    Matrix<Type> conjugation(Type (*pf)(const Type&), bool inplace=false);

    Matrix<Type> dot(const Matrix<Type>& matrix);

    Type vector_dot(const Matrix<Type>& matrix);

    Matrix<Type> mul(const Matrix<Type>& matrix);

    Matrix<Type> cross_product(const Matrix<Type>& matrix);

    Type determinant() const;

    Matrix<Type> inverse(bool inplace=false);

    Type row_max(int row) const;

    Type row_min(int row) const;

    Type row_sum(int row) const;

    Type row_avg(int row) const;

    Type col_max(int col) const;

    Type col_min(int col) const;

    Type col_sum(int col) const;

    Type col_avg(int col) const;

    Matrix<Type> rows_max() const;

    Matrix<Type> cols_max() const;

    Type matrix_max() const;

    Matrix<Type> rows_min() const;

    Matrix<Type> cols_min() const;

    Type matrix_min() const;

    Matrix<Type> rows_sum() const;

    Matrix<Type> cols_sum() const;

    Type matrix_sum() const;

    Matrix<Type> rows_avg() const;

    Matrix<Type> cols_avg() const;

    Type matrix_avg() const;

    Type trace() const;

    Matrix<Type> slice(pair<int, int> rows, pair<int, int> cols, int rows_step=1, int cols_step=1) const;

    int rank() const;

    string getMatrixType() const;

    Matrix<Type> convolution(const Matrix<Type>& kernel, int stride=1, int padding=0);

    void copyToMatrix(Matrix<Type>& matrix) const;

    vector<Type> getEigenValues() const;

    Matrix<Type> convertToHessenbergMatrix() const;

    Matrix<Type> getEigenVectors() const;

    Type norm(int order = 2);

    friend ostream& operator<<(ostream& os, const Matrix<Type>& matrix){
        cout<<"Matrix Print:"<<endl;
        if (matrix.matrixType == 0) {
            for (int i = 0; i < matrix.rows(); ++i) {
                for (int j = 0; j < matrix.cols(); ++j) {
                    cout << matrix.data[i][j] << "\t";
                }
                cout << endl;
            }
        }
        else {
            for (auto it = matrix.data_sparse.begin(); it != matrix.data_sparse.end(); it++) {
                pair<int, int> location = it->first;
                Type value = it->second;
                cout << "(" << location.first << "," << location.second << ") : " << value << endl;
            }
        }
        return os;
    }

    static Mat MatrixToMat_Float(const Matrix<float>& matrix);
    static Mat MatrixToMat_Double(const Matrix<double>& matrix);
    static Mat MatrixToMat_Int(const Matrix<int>& matrix);
    static Mat MatrixToMat_Uchar(const Matrix<uchar>& matrix);

    static Matrix<float> MatToMatrix_Float(const Mat& mat);
    static Matrix<double> MatToMatrix_Double(const Mat& mat);
    static Matrix<int> MatToMatrix_Int(const Mat& mat);
    static Matrix<uchar> MatToMatrix_Uchar(const Mat& mat);
    
    static  Matrix<Type> solveLinearEquations(const Matrix<Type>& A, const Matrix<Type>& b);

    static Matrix<Type> Guess(const Matrix<Type>& matrix);
    static Matrix<Type> Guess(const Matrix<Type>& matrix, int*& outOrder);

    static Matrix<Type> DenseToSparse(Matrix<Type>& dense_matrix, bool inplace = false);
    static Matrix<Type> SparseToDense(Matrix<Type>& sparse_matrix, bool inplace = false);

private:

    vector<vector<Type>> data;

    int matrixType;
    
    map<pair<int,int>,Type> data_sparse;

    static class HessenbergHouseholderUtils{
    public:
        static Matrix<Type> getHouseholderAtCol(Matrix<Type>& R, int col){
            Matrix<Type> I = eyes(R.rows());
            Matrix<Type> P = getMatrixPAtSpecificCol(R, col);
            Matrix<Type> double_P = P * static_cast<Type>(2);
            return (I - double_P);
        }

        static Matrix<Type> getMatrixPAtSpecificCol(Matrix<Type>& R, int col){
            Matrix<Type> v = getMatrixVAtSpecificCol(R, col);
            Matrix<Type> vt = v.transposition();
            Matrix<Type> v_mul_vt = v * vt;
            Matrix<Type> vt_mul_v = vt * v;
            Matrix<Type> P = v_mul_vt * (1/vt_mul_v.data[0][0]);
            return P;
        }

        static Matrix<Type> getMatrixVAtSpecificCol(Matrix<Type>& R, int col){
            vector<Type> x = getXAtSpecificCol(R, col);
            vector<Type> w = getWAtSpecificCol(x, col);
            vector<Type> v = getVAtSpecificCol(x, w);

            Matrix<Type> pv = Matrix<Type>(v.size(), 1);
            for (int i = 0; i < pv.rows(); ++i) {
                pv.data[i][0] = v[i];
            }
            return pv;
        }

        static vector<Type> getXAtSpecificCol(Matrix<Type>& R, int col){
            vector<Type> x(R.rows());
            for (int i = 0; i < R.rows(); ++i) {
                if (i <= col) x[i] = 0;
                else x[i] = R.data[i][col];
            }
            return x;
        }
        static vector<Type> getWAtSpecificCol(vector<Type>& x, int col){
            vector<Type> w(x.size());
            Type lengthOfVector = 0;
            for (int i = 0; i < x.size(); ++i) {
                lengthOfVector += pow(x[i], 2);
            }
            lengthOfVector = sqrt(lengthOfVector);
            w[col+1] = lengthOfVector;
            for (int i = 0; i < x.size(); ++i) {
                if (i != col+1) w[i] = 0;
            }
            return w;
        }
        static vector<Type> getVAtSpecificCol(vector<Type>& x, vector<Type>& w){
            vector<Type> v(w.size());
            int sign = 1;
            if(x[0] > 0) sign *= -1;
            for (int i = 0; i < w.size(); ++i) {
                v[i] = w[i] + sign * x[i];
            }
            return v;
        }
    }HessenbergHouseholderUtilsInstance;

    static Matrix<Type> getRForAStep(Matrix<Type> R, int i);

    static bool isConvergence(Matrix<Type>& matrix);
};


template <typename Type>
class Tensor :public AbstractTensor<Type> {
    Type* data;
    vector<int> size;
public:
    Tensor(vector<int> size) {
        int total = 1;
        this->size = size;
        for (int i = 0; i < size.size(); i++) {
            total *= size[i];
        }
        data = new Type[total];
    }
    Type get(vector<int> where);
    void set(vector<int> where, Type value);
    Tensor<Type> operator +(const Tensor<Type>& another);
    Tensor<Type> operator -(const Tensor<Type>& another);
    Tensor<Type>& operator=(const Tensor<Type>& another);
    ~Tensor() = default;
};


template <typename Type>
Matrix<Type>::Matrix(int rows, int cols) {
    if (rows<=0 || cols<=0) throw MatrixShapeNotValidException(rows, cols);
    this->data = vector<vector<Type>>(rows, vector<Type>(cols));
    this->matrixType = 0;
}

template <typename Type>
Matrix<Type>::Matrix(int rows, int cols, Type val) {
    if (rows<=0 || cols<=0) throw MatrixShapeNotValidException(rows, cols);
    this->data = vector<vector<Type>>(rows, vector<Type>(cols, val));
    this->matrixType = 0;
}

template <typename Type>
Matrix<Type>::Matrix(const Matrix<Type> &matrix) {
    this->data = vector<vector<Type>>(matrix.data);
    this->matrixType = matrix.matrixType;
    this->data_sparse = matrix.data_sparse;
}

template <typename Type>
Matrix<Type>::Matrix(const vector<vector<Type>> &vec) {
    this->data = vector<vector<Type>>(vec);
    this->matrixType = 0;
}

template <typename Type>
Matrix<Type>::Matrix(map<pair<int, int>, Type>& data_sparse) {
    this->data_sparse = data_sparse;
    this->matrixType = 1;
}

template <typename Type>
Matrix<Type>::~Matrix() = default;

template <typename Type>
Matrix<Type> & Matrix<Type>::operator=(const Matrix<Type> &matrix) {
    if (this!=&matrix){
        this->data = vector<vector<Type>>(matrix.data);
        this->matrixType = matrix.matrixType;
        this->data_sparse = matrix.data_sparse;
    }
    return *this;
}

template <typename Type>
Matrix<Type> Matrix<Type>::zeros(int rows, int cols) {
    return Matrix<Type>(rows, cols, static_cast<Type>(0));
}

template <typename Type>
Matrix<Type> Matrix<Type>::ones(int rows, int cols) {
    return Matrix<Type>(rows, cols, static_cast<Type>(1));
}

template <typename Type>
Matrix<Type> Matrix<Type>::values(int rows, int cols, Type val) {
    return Matrix<Type>(rows, cols, val);
}

template <typename Type>
Matrix<Type> Matrix<Type>::eyes(int n, int val) {
    Matrix<Type> matrix(n, n);
    for (int i = 0; i < n; ++i) {
        matrix.data[i][i] = val;
    }
    return matrix;
}

template <typename Type>
int Matrix<Type>::rows() const {
    if (this->matrixType != 0)
    {
        throw MatrixNotDenseException();
    }
    return this->data.size();
}

template <typename Type>
int Matrix<Type>::cols() const {
    if (this->matrixType != 0)
    {
        throw MatrixNotDenseException();
    }
    return this->data.front().size();
}

template <typename Type>
pair<int, int> Matrix<Type>::shape() const {
    return {rows(), cols()};
}

template <typename Type>
int Matrix<Type>::size() const {
    return this->rows()*this->cols();
}

template <typename Type>
bool Matrix<Type>::is_square() const {
    return rows() == cols();
}

template <typename Type>
bool Matrix<Type>::is_vector() const {
    return this->rows()==1 || this->cols()==1;
}

template <typename Type>
Matrix<Type> Matrix<Type>::reshape(int rows, int cols, bool inplace) {
    if (this->cols()*this->rows() != rows*cols){
        throw MatrixReshapeNotValidException(this->size(), rows*cols);
    }
    vector<Type> all_data;
    for (int i = 0; i < this->rows(); ++i) {
        for (int j = 0; j < this->cols(); ++j) {
            all_data.push_back(this->data[i][j]);
        }
    }
    Matrix<Type> result(rows, cols);
    for (int i = 0, cnt = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result.data[i][j] = all_data[cnt++];
        }
    }
    if (inplace){
        this->data = result.data;
    }
    return result;
}

template <typename Type>
Matrix<Type> Matrix<Type>::operator+(const Type &val) {
    Matrix<Type> result(*this);
    for (int i = 0; i < result.rows(); ++i) {
        for (int j = 0; j < result.cols(); ++j) {
            result.data[i][j] = result.data[i][j] + val;
        }
    }
    return result;
}

template <typename Type>
Matrix<Type> Matrix<Type>::operator+(const Matrix<Type> &matrix) {
    Matrix<Type> result(*this);
    for (int i = 0; i < result.rows(); ++i) {
        for (int j = 0; j < result.cols(); ++j) {
            result.data[i][j] = result.data[i][j] + matrix.data[i][j];
        }
    }
    return result;
}

template <typename Type>
Matrix<Type> Matrix<Type>::operator-(const Type &val) {
    Matrix<Type> result(*this);
    for (int i = 0; i < result.rows(); ++i) {
        for (int j = 0; j < result.cols(); ++j) {
            result.data[i][j] = result.data[i][j] - val;
        }
    }
    return result;
}

template <typename Type>
Matrix<Type> Matrix<Type>::operator-(const Matrix<Type> &matrix) {
    Matrix<Type> result(*this);
    for (int i = 0; i < result.rows(); ++i) {
        for (int j = 0; j < result.cols(); ++j) {
            result.data[i][j] = result.data[i][j] - matrix.data[i][j];
        }
    }
    return result;
}

template <typename Type>
Matrix<Type> Matrix<Type>::operator/(const Type &val) {
    Matrix<Type> result(*this);
    for (int i = 0; i < result.rows(); ++i) {
        for (int j = 0; j < result.cols(); ++j) {
            result.data[i][j] = result.data[i][j] / val;
        }
    }
    return result;
}

template <typename Type>
Matrix<Type> Matrix<Type>::operator*(const Type &val) {
    Matrix<Type> result(*this);
    for (int i = 0; i < result.rows(); ++i) {
        for (int j = 0; j < result.cols(); ++j) {
            result.data[i][j] = result.data[i][j] * val;
        }
    }
    return result;
}

template <typename Type>
Matrix<Type> Matrix<Type>::operator*(const Matrix<Type> &matrix) {
    if (matrix.size() == 1) return *this * matrix.data.front().front();
    if (this->cols()!=matrix.rows()) throw MatrixMultiplicationShapeNotValidException(this->shape(), matrix.shape());
    Matrix<Type> result(this->rows(), matrix.cols());
    for (int i = 0; i < this->rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            for (int k = 0; k < this->cols(); ++k) {
                result.data[i][j] = result.data[i][j] + this->data[i][k] * matrix.data[k][j];
            }
        }
    }
    return result;
}

template <typename Type>
Matrix<Type> Matrix<Type>::transposition(bool inplace) {
    Matrix<Type> result(this->cols(), this->rows());
    for (int i = 0; i < this->cols(); ++i) {
        for (int j = 0; j < this->rows(); ++j) {
            result.data[i][j] = this->data[j][i];
        }
    }
    if (inplace){
        this->data = result.data;
    }
    return result;
}

template <typename Type>
Matrix<Type> Matrix<Type>::conjugation(bool inplace) {
    Matrix<Type> result(*this);
    for (int i = 0; i < result.rows(); ++i) {
        for (int j = 0; j < result.cols(); ++j) {
            result.data[i][j] = conj(result.data[i][j]);
        }
    }
    if (inplace){
        this->data = result.data;
    }
    return result;
}

template <typename Type>
Matrix<Type> Matrix<Type>::conjugation(Type (*pf)(const Type & ), bool inplace) {
    if (pf == nullptr){
        throw DefinedConjugationFunctionNotValidException(typeid(Type).name());
    }
    Matrix<Type> result(*this);
    for (int i = 0; i < result.rows(); ++i) {
        for (int j = 0; j < result.cols(); ++j) {
            try{
                result.data[i][j] = (*pf)(result.data[i][j]);
            }catch (exception& e){
                throw DefinedConjugationFunctionNotValidException(typeid(Type).name());
            }
        }
    }
    if (inplace){
        this->data = result.data;
    }
    return result;
}

template <typename Type>
Matrix<Type> Matrix<Type>::dot(const Matrix<Type> &matrix) {
    if (this->shape() != matrix.shape()){
        throw DotMatrixShapeNotValid(this->shape(), matrix.shape());
    }
    Matrix<Type> result(this->rows(), this->cols());
    for (int i = 0; i < result.rows(); ++i) {
        for (int j = 0; j < result.cols(); ++j) {
            result.data[i][j] = this->data[i][j] * matrix.data[i][j];
        }
    }
    return result;
}

template <typename Type>
Type Matrix<Type>::vector_dot(const Matrix<Type>& matrix) {
    Type result(0);
    int n = max(this->cols(), this->rows());
    Matrix<Type> v_1 = Matrix<Type>(*this).reshape(1, n);
    Matrix<Type> v_2 = Matrix<Type>(matrix).reshape(1, n);
    for (int i = 0; i < v_1.cols(); ++i) {
        result = result + v_1.data[0][i] * v_2.data[0][i];
    }
    return result;
}

template <typename Type>
Matrix<Type> Matrix<Type>::mul(const Matrix<Type> &matrix) {
    return *this * matrix;
}

template <typename Type>
Matrix<Type> Matrix<Type>::cross_product(const Matrix<Type> &matrix) {
    if (!this->is_vector() || this->size()!=3){
        throw CrossProductVectorNotValidException(this->shape());
    }
    if (!matrix.is_vector() || matrix.size()!=3){
        throw CrossProductVectorNotValidException(matrix.shape());
    }
    // normal both to row vector
    Matrix<Type> v_1 = this->reshape(1, this->size());
    Matrix<Type> v_2 = Matrix<Type>(matrix).reshape(1, matrix.size());

    Matrix<Type> result(1, 3);
    result.data[0][0] = v_1.data[0][1]*v_2.data[0][2] - v_1.data[0][2]*v_2.data[0][1];
    result.data[0][1] = v_1.data[0][2]*v_2.data[0][0] - v_1.data[0][0]*v_2.data[0][2];
    result.data[0][2] = v_1.data[0][0]*v_2.data[0][1] - v_1.data[0][1]*v_2.data[0][0];
    return result;
}

template <typename Type>
Type Matrix<Type>::determinant() const {
    if (this->size()==1) return this->data[0][0];
    if (!this->is_square()) throw DeterminantMatrixNotSquareException(this->rows(), this->cols());
    int n = this->rows();
    Type result(0);
    Matrix<Type> sub_matrix(n-1, n-1);
    // expand by the first column
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n-1; ++j) {
            for (int k = 0; k < n-1; ++k) {
                sub_matrix.data[j][k] = this->data[(j<i?j:j+1)][k+1];
            }
        }
        result = result + this->data[i][0] * (i%2==0?1:-1) * sub_matrix.determinant();
    }
    return result;
}

template <typename Type>
Matrix<Type> Matrix<Type>::inverse(bool inplace) {
    if (!this->is_square()) throw MatrixInverseNotValidException("matrix is not square");
    if (this->determinant()==0) throw MatrixInverseNotValidException("matrix is a singular matrix with determinant as 0");
    if (this->size()==1) return Matrix<Type>(vector<vector<Type>>(1,vector<Type>(1, 1/(this->data.front().front()))));
    int n = this->rows();
    Type det = this->determinant();
    Matrix<Type> result(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            vector<vector<Type>> temp = this->data;

            // delete the j column
            for (int k = 0; k < temp.size(); ++k) {
                temp[k].erase(temp[k].begin()+j);
            }
            // delete the i row
            temp.erase(temp.begin()+i);
            result.data[i][j] = (((i+j)%2==0)?1:-1) * Matrix<Type>(temp).determinant();
        }
    }
    result = result.transposition();
    result = result / det;
    return result;
}

template <typename Type>
Type Matrix<Type>::row_max(int row) const {
    if (row<1 || row>(this->rows())) throw MatrixIndexNotValidException();
    const vector<Type>& row_vector = this->data[row-1];
    return *(max_element(row_vector.begin(), row_vector.end()));
}

template <typename Type>
Type Matrix<Type>::row_min(int row) const {
    if (row<1 || row>(this->rows())) throw MatrixIndexNotValidException();
    const vector<Type>& row_vector = this->data[row-1];
    return *min_element(row_vector.begin(), row_vector.end());
}

template <typename Type>
Type Matrix<Type>::row_sum(int row) const {
    if (row<1 || row>(this->rows())) throw MatrixIndexNotValidException();
    const vector<Type>& row_vector = this->data[row-1];
    return accumulate(row_vector.begin(), row_vector.end(), static_cast<Type>(0));
}

template <typename Type>
Type Matrix<Type>::row_avg(int row) const {
    if (row<1 || row>(this->rows())) throw MatrixIndexNotValidException();
    const vector<Type>& row_vector = this->data[row-1];
    return accumulate(row_vector.begin(), row_vector.end(), static_cast<Type>(0)) / row_vector.size();
}

template <typename Type>
Type Matrix<Type>::col_max(int col) const {
    if (col<1 || col>(this->cols())) throw MatrixIndexNotValidException();
    vector<Type> col_vector;
    for (int i = 0; i < this->rows(); ++i) {
        col_vector.push_back(this->data[i][col-1]);
    }
    return *max_element(col_vector.begin(), col_vector.end());
}

template <typename Type>
Type Matrix<Type>::col_min(int col) const {
    if (col<1 || col>(this->cols())) throw MatrixIndexNotValidException();
    vector<Type> col_vector;
    for (int i = 0; i < this->rows(); ++i) {
        col_vector.push_back(this->data[i][col-1]);
    }
    return *min_element(col_vector.begin(), col_vector.end());
}

template <typename Type>
Type Matrix<Type>::col_sum(int col) const {
    if (col<1 || col>(this->cols())) throw MatrixIndexNotValidException();
    vector<Type> col_vector;
    for (int i = 0; i < this->rows(); ++i) {
        col_vector.push_back(this->data[i][col-1]);
    }
    return accumulate(col_vector.begin(), col_vector.end(), static_cast<Type>(0));
}

template <typename Type>
Type Matrix<Type>::col_avg(int col) const {
    if (col<1 || col>(this->cols())) throw MatrixIndexNotValidException();
    vector<Type> col_vector;
    for (int i = 0; i < this->rows(); ++i) {
        col_vector.push_back(this->data[i][col-1]);
    }
    return accumulate(col_vector.begin(), col_vector.end(), static_cast<Type>(0)) / col_vector.size();
}

template <typename Type>
Matrix<Type> Matrix<Type>::rows_max() const {
    Matrix<Type> result(this->rows(), 1);
    for (int i = 0; i < this->rows(); ++i) {
        result.data[i][0] = row_max(i+1);
    }
    return result;
}

template <typename Type>
Matrix<Type> Matrix<Type>::rows_min() const {
    Matrix<Type> result(this->rows(), 1);
    for (int i = 0; i < this->rows(); ++i) {
        result.data[i][0] = row_min(i+1);
    }
    return result;
}

template <typename Type>
Matrix<Type> Matrix<Type>::rows_sum() const {
    Matrix<Type> result(this->rows(), 1);
    for (int i = 0; i < this->rows(); ++i) {
        result.data[i][0] = row_sum(i+1);
    }
    return result;
}

template <typename Type>
Matrix<Type> Matrix<Type>::rows_avg() const{
    Matrix<Type> result(this->rows(), 1);
    for (int i = 0; i < this->rows(); ++i) {
        result.data[i][0] = row_avg(i+1);
    }
    return result;
}

template <typename Type>
Matrix<Type> Matrix<Type>::cols_max() const {
    Matrix<Type> result(1, this->cols());
    for (int i = 0; i < this->cols(); ++i) {
        result.data[0][i] = col_max(i+1);
    }
    return result;
}

template <typename Type>
Matrix<Type> Matrix<Type>::cols_min() const {
    Matrix<Type> result(1, this->cols());
    for (int i = 0; i < this->cols(); ++i) {
        result.data[0][i] = col_min(i+1);
    }
    return result;
}

template <typename Type>
Matrix<Type> Matrix<Type>::cols_sum() const {
    Matrix<Type> result(1, this->cols());
    for (int i = 0; i < this->cols(); ++i) {
        result.data[0][i] = col_sum(i+1);
    }
    return result;
}

template <typename Type>
Matrix<Type> Matrix<Type>::cols_avg() const {
    Matrix<Type> result(1, this->cols());
    for (int i = 0; i < this->cols(); ++i) {
        result.data[0][i] = col_avg(i+1);
    }
    return result;
}

template <typename Type>
Type Matrix<Type>::matrix_max() const {
    vector<Type> colsMax_vector = cols_max().data.front();
    return max_element(colsMax_vector.begin(), colsMax_vector.end());
}

template <typename Type>
Type Matrix<Type>::matrix_min() const {
    vector<Type> colsMin_vector = cols_min().data.front();
    return max_element(colsMin_vector.begin(), colsMin_vector.end());
}

template <typename Type>
Type Matrix<Type>::matrix_sum() const {
    vector<Type> colsSum_vector = cols_sum().data.front();
    return accumulate(colsSum_vector.begin(), colsSum_vector.end(), static_cast<Type>(0));
}

template <typename Type>
Type Matrix<Type>::matrix_avg() const {
    return this->matrix_sum() / this->size();
}

template <typename Type>
Type Matrix<Type>::trace() const {
    if(!this->is_square()) throw MatrixTraceNotSquareException(this->rows(), this->cols());
    Type result(0);
    for (int i = 0; i < this->rows(); ++i) {
        result = result + this->data[i][i];
    }
    return result;
}

template <typename Type>
Matrix<Type> Matrix<Type>::slice(pair<int, int> rows, pair<int, int> cols, int rows_step, int cols_step) const {
    if (rows.first<1||rows.second>this->rows()||cols.first<1||cols.second>this->cols()) throw MatrixIndexNotValidException();
    vector<vector<Type>> vec;
    for (int i = rows.first - 1; i < this->rows() && i<rows.second ; i+=rows_step) {
        vec.push_back(vector<Type>());
        for (int j = cols.first - 1; j < this->cols() && j<cols.second; j+=cols_step) {
            vec.back().push_back(this->data[i][j]);
        }
    }
    return Matrix<Type>(vec);
}

template <typename Type>
void Matrix<Type>::copyToMatrix(Matrix<Type> &matrix) const {
    if (this->shape() != matrix.shape()) throw MatrixShapeNotSameException();
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            matrix.data[i][j] = this->data[i][j];
        }
    }
}

template <typename Type>
Matrix<Type> Matrix<Type>::convertToHessenbergMatrix() const {
    int rows = this->rows();
    int cols = this->cols();
    Matrix<Type> R = Matrix<Type>(rows, cols);
    this->copyToMatrix(R);
    for (int i = 0; i < rows - 2; ++i) {
        Matrix<Type> H = HessenbergHouseholderUtilsInstance.getHouseholderAtCol(R, i);
        if (H.rows() == 0) return R;
        R = H * R;
        R = R * H;
    }
    return R;
}

template <typename Type>
Matrix<Type> Matrix<Type>::getRForAStep(Matrix<Type> R, int i) {
    Matrix<Type> partR = eyes(R.rows());
    Type r = sqrt(R.data[i][i]*R.data[i][i] + R.data[i+1][i]*R.data[i+1][i]);
    partR.data[i][i] = R.data[i][i] / r;
    partR.data[i][i+1] = R.data[i+1][i] / r;
    partR.data[i+1][i+1] = R.data[i][i] / r;
    partR.data[i+1][i] = -1 * R.data[i+1][i] / r;
    return partR;
}

template <typename Type>
bool Matrix<Type>::isConvergence(Matrix<Type> &matrix) {
    const Type esp = static_cast<Type>(0.0001);
    for (int i = 1; i < matrix.rows(); ++i) {
        if (abs(matrix.data[i][i-1]) >= esp){
            return false;
        }
    }
    return true;
}

template <typename Type>
vector<Type> Matrix<Type>::getEigenValues() const {
    Matrix<Type> hessenberg = this->convertToHessenbergMatrix();
    int rows = hessenberg.rows();
    int cols = hessenberg.cols();
    Matrix<Type> B = Matrix<Type>(rows, cols);
    hessenberg.copyToMatrix(B);
    while (true){
        Matrix<Type> Q = eyes(rows);
        Matrix<Type> R = Matrix<Type>(rows, cols);
        B.copyToMatrix(R);
        for (int i = 0; i < rows-1; ++i) {
            Matrix<Type> partR = getRForAStep(R, i);
            R = partR * R;
            Matrix<Type> RT = partR.transposition();
            Q = Q * RT;
        }
        B = R * Q;
        if (isConvergence(B)){
            vector<Type> b(cols, 0);
            vector<Type> eigenValues(B.rows());
            for (int i = 0; i < B.rows(); ++i) {
                eigenValues[i] = B.data[i][i];
            }
            return eigenValues;
        }
    }
}

template <typename Type>
Matrix<Type> Matrix<Type>::convolution(const Matrix<Type>& kernel, int stride, int padding) {
    int kernel_rows = kernel.rows();
    int kernel_cols = kernel.cols();
    int padding_rows = this->rows() + 2 * padding;
    int padding_cols = this->cols() + 2 * padding;
    int new_rows = (this->rows() + 2 * padding - kernel_rows) / stride + 1;
    int new_cols = (this->cols() + 2 * padding - kernel_cols) / stride + 1;
    Matrix<Type> padding_matrix(padding_rows, padding_cols);
    Matrix<Type> result_matrix(new_rows, new_cols);
    for (int i = 0; i < this->rows(); ++i) {
        for (int j = 0; j < this->cols(); ++j) {
            padding_matrix.data[i + padding][j + padding] = this->data[i][j];
        }
    }
    queue<Type> q;
    for (int i = 0; i < padding_rows; i += stride) {
        for (int j = 0; j < padding_cols; j += stride) {
            if (i + kernel_rows - 1 < padding_rows && j + kernel_cols - 1 < padding_cols) {
                Type conv_value(0);
                for (int k = 0; k < kernel_rows; ++k) {
                    for (int l = 0; l < kernel_cols; ++l) {
                        conv_value = conv_value + kernel.data[k][l] * padding_matrix.data[k + i][l + j];
                    }
                }
                q.push(conv_value);
            }
        }
    }
    if (q.size() != new_rows * new_cols) throw MatrixConvolutionNotValidException();
    for (int i = 0; i < new_rows; ++i) {
        for (int j = 0; j < new_cols; ++j) {
            result_matrix.data[i][j] = q.front();
            q.pop();
        }
    }
    return result_matrix;
}

template <typename Type>
Type Matrix<Type>::norm(int order) {
    Type result(0);
    if (order == 1) {
        for (int i = 0; i < this->rows(); ++i) {
            for (int j = 0; j < this->cols(); ++j) {
                result = result + abs(this->data[i][j]);
            }
        }
    }
    else if (order == 2) {
        for (int i = 0; i < this->rows(); ++i) {
            for (int j = 0; j < this->cols(); ++j) {
                result = result + this->data[i][j] * this->data[i][j];
            }
        }
        result = sqrt(result);
    }
    else {
        for (int i = 0; i < this->rows(); ++i) {
            for (int j = 0; j < this->cols(); ++j) {
                if (abs(this->data[i][j]) > abs(result)) {
                    result = abs(this->data[i][j]);
                }
            }
        }
    }
    return result;
}

template <typename Type>
Matrix<Type> Matrix<Type>::getEigenVectors() const {
    const int N = 4;
    vector<Type> eigenValues = getEigenValues();
    vector<Matrix<Type>> eigenVectors;
    Matrix<Type> A(*this);
    for (int i = 0; i < eigenValues.size(); ++i) {
        Matrix<Type> eigenVector_0(this->rows(), 1, static_cast<Type>(1));
        Matrix<Type> eigenVector_1(this->rows(), 1, static_cast<Type>(2));
        eigenVector_1 = eigenVector_1 / eigenVector_1.norm();
        Type lambda = eigenValues[i];
        for (int j = 0; j < N; ++j) {
            eigenVector_0 = eigenVector_1;
            eigenVector_1 = (A - eyes(this->rows(), lambda)).inverse() * eigenVector_0;
            eigenVector_1 = eigenVector_1 / eigenVector_1.norm();
        }
        eigenVectors.push_back(eigenVector_1);
    }

    Matrix<Type> result(this->rows(), eigenValues.size());
    for (int j = 0; j < result.cols(); ++j) {
        for (int i = 0; i < result.rows(); ++i) {
            result.data[i][j] = eigenVectors[j].data[i][0];
        }
    }
    return result;
}

template <typename Type>
Mat Matrix<Type>::MatrixToMat_Float(const Matrix<float>& matrix) {
    Mat mat(matrix.rows(), matrix.cols(), CV_32F, Scalar(0));
    for (int i = 0; i < matrix.rows(); i++)
    {
        for (int j = 0; j < matrix.cols(); j++)
        {
            mat.at<float>(i, j) = matrix.data[i][j];
        }
    }
    return mat;
}

template <typename Type>
Mat Matrix<Type>::MatrixToMat_Double(const Matrix<double>& matrix) {
    Mat mat(matrix.rows(), matrix.cols(), CV_64F, Scalar(0));
    for (int i = 0; i < matrix.rows(); i++)
    {
        for (int j = 0; j < matrix.cols(); j++)
        {
            mat.at<double>(i, j) = matrix.data[i][j];
        }
    }
    return mat;
}

template <typename Type>
Mat Matrix<Type>::MatrixToMat_Int(const Matrix<int>& matrix) {
    Mat mat(matrix.rows(), matrix.cols(), CV_32S, Scalar(0));
    for (int i = 0; i < matrix.rows(); i++)
    {
        for (int j = 0; j < matrix.cols(); j++)
        {
            mat.at<int>(i, j) = matrix.data[i][j];
        }
    }
    return mat;
}

template <typename Type>
Mat Matrix<Type>::MatrixToMat_Uchar(const Matrix<uchar>& matrix) {
    Mat mat(matrix.rows(), matrix.cols(), CV_8U, Scalar(0));
    for (int i = 0; i < matrix.rows(); i++)
    {
        for (int j = 0; j < matrix.cols(); j++)
        {
            mat.at<uchar>(i, j) = matrix.data[i][j];
        }
    }
    return mat;
}

template <typename Type>
string Matrix<Type>::getMatrixType() const {
    if (this->matrixType == 0)
    {
        return "Dense Matrix";
    }
    else
    {
        return "Sparse Matrix";
    }
}

template <typename Type>
Matrix<float> Matrix<Type>::MatToMatrix_Float(const Mat& mat) {
    Matrix<float> matrix(mat.rows, mat.cols);
    for (int i = 0; i < mat.rows; i++)
    {
        for (int j = 0; j < mat.cols; j++) {
            matrix.data[i][j] = mat.at<float>(i, j);
        }
    }
    return matrix;
}

template <typename Type>
Matrix<double> Matrix<Type>::MatToMatrix_Double(const Mat& mat) {
    Matrix<double> matrix(mat.rows, mat.cols);
    for (int i = 0; i < mat.rows; i++)
    {
        for (int j = 0; j < mat.cols; j++) {
            matrix.data[i][j] = mat.at<double>(i, j);
        }
    }
    return matrix;
}

template <typename Type>
Matrix<int> Matrix<Type>::MatToMatrix_Int(const Mat& mat) {
    Matrix<int> matrix(mat.rows, mat.cols);
    for (int i = 0; i < mat.rows; i++)
    {
        for (int j = 0; j < mat.cols; j++) {
            matrix.data[i][j] = mat.at<int>(i, j);
        }
    }
    return matrix;
}

template <typename Type>
Matrix<uchar> Matrix<Type>::MatToMatrix_Uchar(const Mat& mat) {
    Matrix<uchar> matrix(mat.rows, mat.cols);
    for (int i = 0; i < mat.rows; i++)
    {
        for (int j = 0; j < mat.cols; j++) {
            matrix.data[i][j] = mat.at<uchar>(i, j);
        }
    }
    return matrix;
}

template <typename Type>
int Matrix<Type>::rank()const {
    const int rowNum = rows(), columnNum = cols();
    int i, j, k, loc, colInd;
    const double eps = 1e-8;
    Type maxElement, ele, ratio;
    vector<vector<Type>>matrix(rowNum, vector<Type>(columnNum));
    for (i = 0; i < rowNum; i++) {
        for (j = 0; j < columnNum; j++) {
            matrix[i][j] = data[i][j];
        }
    }
    int whichRow = 0, whichColumn = 0;
    for (i = 0; whichRow < rowNum && whichColumn < columnNum; i++) {
        int cp = i;
        i = whichRow;
        maxElement = matrix[i][whichColumn];
        loc = i;
        for (j = i + 1; j < rowNum; j++) {
            if (abs(maxElement) < abs(matrix[j][whichColumn])) {
                maxElement = matrix[j][whichColumn];
                loc = j;
            }
        }
        if (abs(maxElement) < static_cast<Type>(eps)) {
            whichColumn++;
            i = cp;
            continue;
        }
        for (j = 0; j < columnNum; j++) {
            ele = matrix[loc][j];
            matrix[loc][j] = matrix[i][j];
            matrix[i][j] = ele;
        }
        for (j = i + 1; j < rowNum; j++) {
            ratio = matrix[j][whichColumn] / maxElement;
            for (k = 0; k < columnNum; k++) {
                matrix[j][k] -= ratio * matrix[i][k];
            }
        }
        i = cp;
        whichRow++;
        whichColumn++;
        if (whichRow >= rowNum) {
            break;
        }
    }
    int result = 0;
    j = 0;
    for (i = 0; i < min(rowNum, columnNum); i++) {
        while ((j < columnNum) &&(abs(matrix[i][j]) < static_cast<Type>(eps))) {
            j++;
        }
        if (j < columnNum) {
            result++;
        }
    }
    return result;
}

template <typename Type>
Matrix<Type> Matrix<Type>::solveLinearEquations(const Matrix<Type>& A, const Matrix<Type>& b) {
    if ((A.rows() != b.rows()) || (b.cols() != 1)) {
        throw  MatrixEquationParameterException();
    }
    int i, j, k, oldRank;
    oldRank = A.rank();
    Matrix<Type> solver = Matrix<Type>(A.rows(), A.cols() + 1);
    for (i = 0; i < A.rows(); i++) {
        for (j = 0; j < A.cols(); j++) {
            solver.data[i][j] = A.data[i][j];
        }
    }
    for (i = 0; i < A.rows(); i++) {
        solver.data[i][A.cols()] = b.data[i][0];
    }
    if (solver.rank() != oldRank) {
        throw  MatrixEquationNoSolutionException();
    }
    int* order;
    if (solver.rank() < solver.rows()) {
        return Guess(solver);
    }
    int rowNum = solver.rows(), columnNum = solver.cols();
    solver = Guess(solver, order);
    vector<vector<Type>>& data = solver.data;
    int cl = columnNum - 2;
    for (i = rowNum - 1; i >= 1; i--) {
        for (j = i - 1; j >= 0; j--) {
            data[j][columnNum - 1] = data[j][columnNum - 1] - data[j][cl] / data[i][cl] * data[i][columnNum - 1];
        }
        cl--;
    }
    Matrix<Type> result = Matrix<Type>(rowNum, 1);
    for (i = 0; i < rowNum; i++) {
        result.data[i][0] = data[i][columnNum - 1] / data[i][i];
    }
    return result;
}

template <typename Type>
Matrix<Type> Matrix<Type>::Guess(const Matrix<Type>& matrix) {
    int i, j, k;
    double eps = 1e-8;
    int rowNum = matrix.rows(), columnNum = matrix.cols(), loc;
    Matrix<Type> returnValue = Matrix<Type>(rowNum, columnNum);
    Type maxElement, ele, ratio;
    vector<vector<Type>>& result = returnValue.data, data = matrix.data;
    vector<int> order(rowNum);
    for (i = 0; i < rowNum; i++) {
        order[i] = i;
    }
    for (i = 0; i < rowNum; i++) {
        for (j = 0; j < columnNum; j++) {
            result[i][j] = data[i][j];
        }
    }
    int whichRow = 0, whichColumn = 0;
    for (i = 0; whichRow < rowNum && whichColumn < columnNum; i++) {
        int cp = i;
        i = whichRow;
        maxElement = result[i][whichColumn];
        loc = i;
        for (j = i + 1; j < rowNum; j++) {
            if (abs(maxElement) < abs(result[j][whichColumn])) {
                maxElement = result[j][whichColumn];
                loc = j;
            }
        }
        if (abs(maxElement) < abs(static_cast<Type>(eps))) {
            i = cp;
            whichColumn++;
            continue;
        }
        swap(order[i], order[loc]);
        for (j = 0; j < columnNum; j++) {
            ele = result[loc][j];
            result[loc][j] = result[i][j];
            result[i][j] = ele;
        }
        for (j = i + 1; j < rowNum; j++) {
            ratio = result[j][whichColumn] / maxElement;
            for (k = 0; k < columnNum; k++) {
                result[j][k] -= ratio * result[i][k];
            }
        }
        whichRow++;
        whichColumn++;
    }

    return returnValue;
}

template <typename Type>
Matrix<Type> Matrix<Type>::Guess(const Matrix<Type>& matrix, int*& outOrder) {
    int i, j, k;
    double eps = 1e-8;
    int rowNum = matrix.rows(), columnNum = matrix.cols(), loc;
    Matrix<Type> returnValue = Matrix<Type>(rowNum, columnNum);
    Type maxElement, ele, ratio;
    vector<vector<Type>>& result = returnValue.data, data = matrix.data;
    int* order = new int[rowNum];
    for (i = 0; i < rowNum; i++) {
        order[i] = i;
    }
    for (i = 0; i < rowNum; i++) {
        for (j = 0; j < columnNum; j++) {
            result[i][j] = data[i][j];
        }
    }
    int whichRow = 0, whichColumn = 0;
    for (i = 0; whichRow < rowNum && whichColumn < columnNum; i++) {
        int cp = i;
        i = whichRow;
        maxElement = result[i][whichColumn];
        loc = i;
        for (j = i + 1; j < rowNum; j++) {
            if (abs(maxElement) < abs(result[j][whichColumn])) {
                maxElement = result[j][whichColumn];
                loc = j;
            }
        }
        if (abs(maxElement) < abs(static_cast<Type>(eps))) {
            i = cp;
            whichColumn++;
            continue;
        }
        swap(order[i], order[loc]);
        for (j = 0; j < columnNum; j++) {
            ele = result[loc][j];
            result[loc][j] = result[i][j];
            result[i][j] = ele;
        }
        for (j = i + 1; j < rowNum; j++) {
            ratio = result[j][whichColumn] / maxElement;
            for (k = 0; k < columnNum; k++) {
                result[j][k] -= ratio * result[i][k];
            }
        }
        whichRow++;
        whichColumn++;
    }
    outOrder = order;
    return returnValue;
}


template <typename Type>
Matrix<Type> Matrix<Type>::DenseToSparse(Matrix<Type>& dense_matrix, bool inplace) {
    if (dense_matrix.matrixType != 0) throw MatrixNotDenseException();
    Matrix<Type> sparseMatrix;
    sparseMatrix.matrixType = 1;
    for (int i = 0; i < dense_matrix.rows(); i++)
    {
        for (int j = 0; j < dense_matrix.cols(); j++)
        {
            if (dense_matrix.data[i][j] != static_cast<Type>(0))
            {
                sparseMatrix.data_sparse[{i, j}] = dense_matrix.data[i][j];
            }
        }
    }
    if (inplace)
    {
        dense_matrix.data.clear();
        dense_matrix.matrixType = 1;
        dense_matrix.data_sparse = sparseMatrix.data_sparse;
    }
    return sparseMatrix;
}

template <typename Type>
Matrix<Type> Matrix<Type>::SparseToDense(Matrix<Type>& sparse_matrix, bool inplace) {
    if (sparse_matrix.matrixType != 1) return MatrixNotSparseException();

    int max_row_index = 0;
    int max_col_index = 0;
    for (auto it = sparse_matrix.data_sparse.begin(); it != sparse_matrix.data_sparse.end(); it++) {
        pair<int, int> location = it->first;
        Type value = it->second;
        max_row_index = max(location.first, max_row_index);
        max_col_index = max(location.second, max_col_index);
    }
    Matrix<Type> denseMatrix(max_row_index + 1, max_col_index + 1);
    denseMatrix.matrixType = 0;
    for (auto it = sparse_matrix.data_sparse.begin(); it != sparse_matrix.data_sparse.end(); it++) {
        pair<int, int> location = it->first;
        Type value = it->second;
        denseMatrix.data[location.first][location.second] = value;
    }
    if (inplace)
    {
        sparse_matrix.data.clear();
        sparse_matrix.matrixType = 0;
        sparse_matrix.data = denseMatrix.data;
    }
    return denseMatrix;
}

template <typename Type>
void Tensor<Type>::set(vector<int> where, Type value) {
    int ind = 0, lax = 1;
    int i, j;
    for (i = where.size() - 1; i >= 0; i--) {
        ind += lax * where[i];
        lax *= size[i];
    }
    data[ind] = value;
}

template <typename Type>
Type Tensor<Type>::get(vector<int> where) {
    int ind = 0, lax = 1;
    int i, j;
    for (i = where.size() - 1; i >= 0; i--) {
        ind += lax * where[i];
        lax *= size[i];
    }
    return data[ind];
}

template <typename Type>
Tensor<Type> Tensor<Type>::operator+(const Tensor<Type>& another) {
    Tensor<Type> result = Tensor<Type>(another.size);
    int total = 1, lax = 1;
    for (int i = result.size.size() - 1; i >= 0; i--) {
        total *= result.size[i];
    }
    for (int i = 0; i < total; i++) {
        result.data[i] = another.data[i] + data[i];
    }
    return result;
}

template <typename Type>
Tensor<Type> Tensor<Type>::operator-(const Tensor<Type>& another) {
    Tensor<Type> result = Tensor<Type>(another.size);
    int total = 1, lax = 1;
    for (int i = result.size.size() - 1; i >= 0; i--) {
        total *= result.size[i];
    }
    for (int i = 0; i < total; i++) {
        result.data[i] = -another.data[i] + data[i];
    }
    return result;
}

template <typename Type>
Tensor<Type>& Tensor<Type>::operator=(const Tensor<Type>& another) {
    int total = 1;
    for (int i = another.size.size() - 1; i >= 0; i--) {
        total *= another.size[i];
    }
    this->data = new Type[total];
    this->size = another.size;
    for (int i = 0; i < total; i++) {
        this->data[i] = another.data[i];
    }
    return *this;
}

class SolveExpressions {
    string expression;
    int bracketEnd[1000];
    map<string, Matrix<double>> match;
public:
    void init(string str) {
        stack<int> st;
        expression = str;
        int i, j, k;
        for (i = 0; i < expression.size(); i++) {
            if (expression[i] == '(') {
                st.push(i);
            }
            else if (expression[i] == ')') {
                bracketEnd[st.top()] = i;
                st.pop();
            }
        }
    }

    Matrix<double> solve(int start, int end) {
        Matrix<double> result;
        int i, j, k, last = start;
        bool isVarible = true;
        vector<Matrix<double>> out;
        stack<int> store, oprSta;
        for (i = start; i <= end; i++) {
            if ((expression[i] == '(') || (expression[i] == ')') || (expression[i] == '=')) {
                isVarible = false;
            }
        }
        if (isVarible) {
            return match.at(expression.substr(start, end - start + 1));
        }
        for (i = start; i <= end; i++) {
            if (expression[i] == '=') {
                match.insert(make_pair(expression.substr(start, i - start), result = getMatrix(i + 2, end - 1)));
                return result;
            }
            if (expression[i] == '(') {
                if (expression.substr(last, i - last) == "Guess") {
                    store.push(out.size());
                    out.push_back(Matrix<double>::Guess(solve(i + 1, bracketEnd[i] - 1)));
                }
                else if (expression.substr(last, i - last) == "rank") {
                    store.push(out.size());
                    out.push_back(Matrix<double>(1, 1, solve(i + 1, bracketEnd[i] - 1).rank()));
                }
                else if (expression.substr(last, i - last) == "inverse") {
                    store.push(out.size());
                    out.push_back(solve(i + 1, bracketEnd[i] - 1).inverse());
                }
                else if (expression.substr(last, i - last) == "determinant") {
                    store.push(out.size());
                    out.push_back(Matrix<double>(1, 1, solve(i + 1, bracketEnd[i] - 1).determinant()));
                }
                else if (expression.substr(last, i - last) == "trace") {
                    store.push(out.size());
                    out.push_back(Matrix<double>(1, 1, solve(i + 1, bracketEnd[i] - 1).trace()));
                }
                else if (expression.substr(last, i - last) == "sum") {
                    store.push(out.size());
                    out.push_back(Matrix<double>(1, 1, solve(i + 1, bracketEnd[i] - 1).matrix_sum()));
                }
                else if (expression.substr(last, i - last) == "transposition") {
                    store.push(out.size());
                    out.push_back(solve(i + 1, bracketEnd[i] - 1).transposition());
                }
                else if (expression.substr(last, i - last) == "rowsMax") {
                    store.push(out.size());
                    out.push_back(solve(i + 1, bracketEnd[i] - 1).rows_max());
                }
                else if (expression.substr(last, i - last) == "rowsMin") {
                    store.push(out.size());
                    out.push_back(solve(i + 1, bracketEnd[i] - 1).rows_min());
                }
                else if (expression.substr(last, i - last) == "colsMax") {
                    store.push(out.size());
                    out.push_back(solve(i + 1, bracketEnd[i] - 1).cols_max());
                }
                else if (expression.substr(last, i - last) == "colsMin") {
                    store.push(out.size());
                    out.push_back(solve(i + 1, bracketEnd[i] - 1).cols_min());
                }
                else if (expression.substr(last, i - last) == "rowsSum") {
                    store.push(out.size());
                    out.push_back(solve(i + 1, bracketEnd[i] - 1).rows_sum());
                }
                else if (expression.substr(last, i - last) == "colsSum") {
                    store.push(out.size());
                    out.push_back(solve(i + 1, bracketEnd[i] - 1).cols_sum());
                }
                else if (expression.substr(last, i - last) == "rowsAvg") {
                    store.push(out.size());
                    out.push_back(solve(i + 1, bracketEnd[i] - 1).rows_avg());
                }
                else if (expression.substr(last, i - last) == "colsAvg") {
                    store.push(out.size());
                    out.push_back(solve(i + 1, bracketEnd[i] - 1).cols_avg());
                }
                else if (expression.substr(last, i - last) == "eigenValues") {
                    store.push(out.size());
                    vector<vector<double>> eigenValues
                    (1, solve(i + 1, bracketEnd[i] - 1).getEigenValues());
                    out.push_back(Matrix<double>(eigenValues));
                }
                else {
                    store.push(out.size());
                    out.push_back(solve(i + 1, bracketEnd[i] - 1));
                }
                i = bracketEnd[i];
                last = i + 1;
            }
            else if ((expression[i] == '+') || (expression[i] == '-') || (expression[i] == '*') || (expression[i] == '/')) {
                int opr;
                switch (expression[i]) {
                case '+':
                    opr = -12;
                    break;
                case '-':
                    opr = -11;
                    break;
                case '*':
                    opr = -1;
                    break;
                case '/':
                    opr = -2;
                    break;
                }
                if (oprSta.empty() || (opr - 5 > oprSta.top())) {
                    oprSta.push(opr);
                }
                else {
                    while ((!oprSta.empty()) && (opr - 5 <= oprSta.top())) {
                        store.push(oprSta.top());
                        oprSta.pop();
                    }
                    oprSta.push(opr);
                }
                last = i + 1;
            }
        }
        while (!oprSta.empty()) {
            store.push(oprSta.top());
            oprSta.pop();
        }
        while (!store.empty()) {
            oprSta.push(store.top());
            store.pop();
        }
        stack<Matrix<double>> wait;
        Matrix<double> matrix1, matrix2;
        while (!oprSta.empty()) {
            if (oprSta.top() >= 0) {
                wait.push(out[oprSta.top()]);
            }
            else {
                switch (oprSta.top()) {
                case -1:
                    matrix1 = wait.top();
                    wait.pop();
                    matrix2 = wait.top();
                    wait.pop();
                    wait.push(matrix2 * matrix1);
                    break;
                case -11:
                    matrix1 = wait.top();
                    wait.pop();
                    matrix2 = wait.top();
                    wait.pop();
                    wait.push(matrix2 - matrix1);
                    break;
                case -12:
                    matrix1 = wait.top();
                    wait.pop();
                    matrix2 = wait.top();
                    wait.pop();
                    wait.push(matrix2 + matrix1);
                    break;
                }
            }
            oprSta.pop();
        }
        matrix1 = wait.top();
        return matrix1;
    }
    Matrix<double> getMatrix(int start, int end) {
        int i, j, k, here = 0, p = 1;
        int row = 1, column = 1;
        bool columnRight = false;
        for (i = start; i <= end; i++) {
            if (!columnRight) {
                if (expression[i] == ',') {
                    column++;
                }
                if (expression[i] == ';') {
                    row++;
                    columnRight = true;
                }
            }
            else {
                if (expression[i] == ';') {
                    row++;
                }

            }
        }
        vector<vector<double>> v(row, vector<double>(column));
        bool negative = false;
        int x = 0, y = 0;
        for (i = start; i <= end; i++) {
            if (expression[i] == ',') {
                v[x][y] = negative ? -here : here;
                here = 0;
                p = 1;
                y++;
                negative = false;
            }
            else if (expression[i] == ';') {
                v[x][y] = negative ? -here : here;
                here = 0;
                p = 1;
                x++;
                y = 0;
                negative = false;
            }
            else if (expression[i] == '-') {
                negative = true;
            }
            else {
                here += (expression[i] - '0') * p;
                p *= 10;
            }
        }
        v[x][y] = negative ? -here : here;;
        Matrix<double> result = Matrix<double>(v);
        return result;
    }
};