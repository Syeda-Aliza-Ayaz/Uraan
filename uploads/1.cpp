#include <iostream>
using namespace std;
int main()
{
    int row, col;
    // input validation for row and column
    while (true)
    {
        cout << "Enter number of rows:"; // assume row=3
        cin >> row;
        cout << "Enter number of col:"; // assume col=4
        cin >> col;
        // row and column must be positive and not 0
        if (row > 0 && col > 0)
        {
            break;
        }
        else
        {
            cout << "Invalid input try again" << endl;
        }
    }
    // Dynamic memory allocation (row pointer)
    int **arr2D = new int *[row]; // as row=3 so we got an array of 3 pointers where each pointer points to one row.
    // arr2d-->[*][*][*] right now these pointer doesn't point anywhere
    // array of row pointers  and each points to a dynamic array of columns
    for (int i = 0; i < row; i++)
    {
        arr2D[i] = new int[col]; // here we got col=4 so now each pointer points to a new array of size 4(4 columns).
        // arr2D[0]->[1 2 3 4]
        // arr2D[1]->[5 6 7 8]
        // arr2D[3]->[9 10 11 12]
    }
    cout << "======Input 2D array======" << endl;
    // Take 2D array input
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            cout << "arr[" << i << "][" << j << "]= ";
            cin >> arr2D[i][j];
        }
    }
    // print 2D array
    cout << "=====2D Array=====" << endl;
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            cout << arr2D[i][j] << " ";
        }
        cout << endl;
    }
    int size = row * col; // size=3*4=12
    int *arr1D = new int[size];
    int idx = 0; // index where element is inserted initially it's 0
    // traversing array column by column
    for (int j = 0; j < col; j++)
    {
        for (int i = 0; i < row; i++)
        {
            arr1D[idx++] = arr2D[i][j]; // copy in 1d array
        }
    }
    // print 1D array in column major order
    cout << "===== Array in column major order ======" << endl;
    for (int i = 0; i < size; i++)
    {
        cout << arr1D[i] << " ";
    }
    for (int i = 0; i < row; i++)
    {
        delete[] arr2D[i]; // delete row
    }
    delete[] arr2D; // delete row pointer array
    delete[] arr1D;
    return 0;
}