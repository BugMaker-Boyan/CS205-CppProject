#include "Structure.hpp"

void test_1();
void test_2();
void test_3();
void test_4();
void test_5();
void test_6();
void test_7();
void test_8();
void test_9();
void test_10();
void test_11();

int main()
{

    test_11();
    
    return 0;
}

void test_1() {
    // create 1 * 1 matrix<int>
    Matrix<int> m_1 = Matrix<int>(1, 1);
    cout << m_1 << endl;

    // create 2*4 matrix<float> with initial value 0.5
    Matrix<float> m_2 = Matrix<float>(2, 4, 0.5);
    cout << m_2 << endl;

    // create 3 * 3 matrix<double> with initial vecator<vector<dobule>>
    vector<vector<double>> vec_1 = vector<vector<double>>(3, vector<double>(3, 0.666));
    Matrix<double> m_3 = Matrix<double>(vec_1);
    cout << m_3 << endl;

    // create sparse matrix<double> with 100000 * 500000
    map<pair<int, int>, double> m;
    m[{0, 1}] = 6.9;
    m[{100000, 500000}] = 9.999;
    Matrix<double> m_4 = Matrix<double>(m);
    cout << m_4 << endl;
}

class user_datatype {
public:
    string value;
    user_datatype(string val) :value(val) {}
};

ostream& operator<<(ostream& os, const user_datatype& v) {
    cout << v.value;
    return os;
}

void test_2() {
    Matrix<char> m_1 = Matrix<char>(2, 2, 'x');
    cout << "Matrix<char>:" << endl;
    cout << m_1 << endl;

    Matrix<short> m_2 = Matrix<short>(2, 2, 2);
    cout << "Matrix<short>:" << endl;
    cout << m_2 << endl;

    Matrix<int> m_3 = Matrix<int>(2, 2, 3);
    cout << "Matrix<int>:" << endl;
    cout << m_3 << endl;

    Matrix<long> m_4 = Matrix<long>(2, 2, 4);
    cout << "Matrix<long>:" << endl;
    cout << m_4 << endl;

    Matrix<float> m_5 = Matrix<float>(2, 2, 5);
    cout << "Matrix<float>:" << endl;
    cout << m_5 << endl;

    Matrix<double> m_6 = Matrix<double>(2, 2, 6);
    cout << "Matrix<double>:" << endl;
    cout << m_6 << endl;

    Matrix<std::complex<double>> m_7 = Matrix<std::complex<double>>(2, 2, {1, 2});
    cout << "Matrix<std::complex<double>>:" << endl;
    cout << m_7 << endl;


    Matrix<user_datatype> m_8 = Matrix<user_datatype>(2, 2, user_datatype("user_datatype"));
    cout << "Matrix<user_datatype>:" << endl;
    cout << m_8 << endl;
}

void test_3() {
    Matrix<double> m_1(3, 3, 1.2);
    Matrix<double> m_2(3, 3, 2.1);
    Matrix<double> m_4(3, 1, 0.5);
    Matrix<complex<double>> m_3(3, 3, { 1,3 });
    double number = 0.5;

    cout << "Matrix<double> m_1:" << endl;
    cout << m_1 << endl;
    cout << "Matrix<double> m_2:" << endl;
    cout << m_2 << endl;
    cout << "Matrix<complex<double>> m_3:" << endl;
    cout << m_3 << endl;
    cout << "Matrix<complex<double>> m_4:" << endl;
    cout << m_4 << endl;
    cout << "double number:" << endl;
    cout << number << endl;


    // test matrix addition
    cout << "(m_1 + m_2)" << endl;
    cout << (m_1 + m_2) << endl;

    // test scalar addition
    cout << "(m_1 + number)" << endl;
    cout << (m_1 + number) << endl;

    // test matrix subtraction
    cout << "(m_1 - m_2)" << endl;
    cout << (m_1 - m_2) << endl;

    // test scalar subtraction
    cout << "(m_1 - number)" << endl;
    cout << (m_1 - number) << endl;

    // test scalar multiplication
    cout << "(m_1 * number)" << endl;
    cout << (m_1 * number) << endl;

    // test matrix multiplication
    cout << "(m_1 * m_2)" << endl;
    cout << (m_1 * m_2) << endl;

    // test scalar division
    cout << "(m_1 / number)" << endl;
    cout << (m_1 / number) << endl;

    // test matrix transposition
    cout << "m_1.transposition()" << endl;
    cout << m_1.transposition() << endl;

    // test matrix conjugation
    cout << "Before conjugation:" << endl;
    cout << m_3 << endl;
    cout << "After conjugation:" << endl;
    cout << m_3.conjugation() << endl;

    // test element-wise multiplication
    cout << "m_1 .* m_2 (element-wise multiplication)" << endl;
    cout << m_1.dot(m_2) << endl;

    // test matrix-vector multiplication
    cout << "m_4 vector-dot m_4" << endl;
    cout << m_4.vector_dot(m_4) << endl;

    // test cross-product
    cout << "m_4 corss-product m_4" << endl;
    cout << m_4.cross_product(m_4) << endl;

}

void test_4() {
    vector<vector<double>> v(3, vector<double>(3));
    v[0][0] = 2;
    v[0][1] = 3;
    v[0][2] = 1;
    v[1][0] = v[1][1] = v[1][2] = 1;
    v[2][0] = 3;
    v[2][1] = 5;
    v[2][2] = 2;
    Matrix<double> m1(v);
    
    cout << "m1.row_max(1)" << endl;
    cout << m1.row_max(1) << endl;
    cout << "m1.col_min(2)" << endl;
    cout << m1.col_min(2) << endl;
    cout << "m1.cols_sum()" << endl;
    cout << m1.cols_sum() << endl;
    cout << "m1.matrix_avg()" << endl;
    cout << m1.matrix_avg() << endl;
}

void test_5() {
    vector<vector<double>> v(3, vector<double>(3));
    v[0][0] = 2;
    v[0][1] = 3;
    v[0][2] = 1;
    v[1][0] = v[1][1] = v[1][2] = 1;
    v[2][0] = 3;
    v[2][1] = 5;
    v[2][2] = 2;
    Matrix<double> m1(v);
    cout << "m1.getEigenValues()" << endl;
    auto ev = m1.getEigenValues();
    for (size_t i = 0; i < ev.size(); i++)
    {
        cout << ev[i] << " ";
    }
    cout << endl;
    cout << "m1.getEigenVectors()" << endl;
    cout << m1.getEigenVectors() << endl;
    cout << "m1.trace()" << endl;
    cout << m1.trace() << endl;
    cout << "m1.inverse()" << endl;
    cout << m1.inverse() << endl;
    cout << "m1.determinant()" << endl;
    cout << m1.determinant() << endl;

}

void test_6() {
    vector<vector<double>> v(3, vector<double>(3));
    v[0][0] = 2;
    v[0][1] = 3;
    v[0][2] = 1;
    v[1][0] = v[1][1] = v[1][2] = 1;
    v[2][0] = 3;
    v[2][1] = 5;
    v[2][2] = 2;
    Matrix<double> m1(v);

    cout << "m1.reshape(1, 9)" << endl;
    cout << m1.reshape(1, 9) << endl;

    cout << "m1.slice({ 1,2 }, { 1,2 })" << endl;
    cout << m1.slice({ 1,2 }, { 1,2 }) << endl;
}

void test_7() {
    vector<vector<double>> v(3, vector<double>(3));
    v[0][0] = 2;
    v[0][1] = 3;
    v[0][2] = 1;
    v[1][0] = v[1][1] = v[1][2] = 1;
    v[2][0] = 3;
    v[2][1] = 5;
    v[2][2] = 2;
    Matrix<double> m1(v);
    vector<vector<double>> v2(2, vector<double>(2));
    v2[0][0] = 1;
    v2[0][1] = 2;
    v2[1][0] = 1;
    v2[1][1] = 1;
    Matrix<double> m2(v2);
    cout << m1.convolution(m2, 1, 0) << endl;
}

void test_8() {
    vector<vector<double>> v(3, vector<double>(3));
    v[0][0] = 2;
    v[0][1] = 3;
    v[0][2] = 1;
    v[1][0] = v[1][1] = v[1][2] = 1;
    v[2][0] = 3;
    v[2][1] = 5;
    v[2][2] = 2;
    Matrix<double> m1(v);
    cout << "Matrix<double>:" << endl;
    cout << m1 << endl;
    Mat m = Matrix<double>::MatrixToMat_Double(m1);
    cout << "Mat:" << endl;
    cout << m << endl;

    Matrix<double> m2 = Matrix<double>::MatToMatrix_Double(m);
    cout << m2 << endl;
}

void test_9() {
    vector<vector<double>> v(3, vector<double>(3));
    v[0][0] = 2;
    v[0][1] = 3;
    v[0][2] = 1;
    v[1][0] = v[1][1] = v[1][2] = 1;
    v[2][0] = 3;
    v[2][1] = 5;
    v[2][2] = 2;
    Matrix<double> m1(v);
    try {
        m1.reshape(1, 1);
    }
    catch (BaseException& e) {
        e.Show();
    }

    Matrix<double> m2(v);
    m2.reshape(1, 9, true);
    try {
        m1* m2;
    }catch(BaseException& e) {
        e.Show();
    }
}

void test_11() {
    vector<vector<double>> v(2, vector<double>(2));
    v[0][0] = 1;
    v[0][1] = 0;
    v[1][0] = 1;
    v[1][1] = 1;
    vector<vector<double>> v1(2, vector<double>(1));
    v1[0][0] = 1;
    v1[1][0] = 2;
    Matrix<double> n1(v), n2(v1);
    try {
        cout << Matrix<double>::solveLinearEquations(n1, n2);
    }
    catch (BaseException& e) {
        e.Show();
    }
}

void test_10() {
    SolveExpressions solver;
    string str;
    while (cin >> str) {
        solver.init(str);
        cout << solver.solve(0, str.size() - 1) << endl;
    }
}