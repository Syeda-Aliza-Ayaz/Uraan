#include <iostream>
#include <algorithm>
using namespace std;
// binary search function
bool binarySearch(int **arr2D, int row, int col, int target)
{
    int size = row * col;
    int left = 0, right = size - 1;
    // loop to traverse through array
    while (left <= right)
    {
        int mid = left + (right - left) / 2; // calculate midpoint
        int r = mid / col;                   // row index
        int c = mid % col;                   // column index
        int midVal = arr2D[r][c];
        if (midVal == target)
        {
            return true;
        }
        else if (midVal < target)
        {
            left = mid + 1;
        }
        else
        {
            right = mid - 1;
        }
    }
    return false;
}
int main()
{
    int row, col;
    // input validation for row and column
    while (true)
    {
        cout << "Enter number of rows:";
        cin >> row;
        cout << "Enter number of col:";
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
    int **arr2D = new int *[row];
    for (int i = 0; i < row; i++)
    {
        arr2D[i] = new int[col]; // now each pointer points to a new array of size n (n columns).
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
    int size = row * col;
    int *arr1D = new int[size];
    int idx = 0; // index where element is inserted initially it's 0
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            arr1D[idx++] = arr2D[i][j]; // copy 2D array to 1D array in order to put sort function
        }
    }
    sort(arr1D, arr1D + size); // sort 1d array
    idx = 0;
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            arr2D[i][j] = arr1D[idx++]; // copy 1D array again to 2D after sorting
        }
    }
    // print 2D array
    cout << "=====Sorted 2D Array=====" << endl;
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            cout << arr2D[i][j] << " "; // print sorted 2D array
        }
        cout << endl;
    }

    int target;
    cout << "Enter target to be searched:";
    cin >> target;
    cout << "===== Binary Search =====" << endl;
    bool result = binarySearch(arr2D, row, col, target);
    cout << boolalpha << result;
    for (int i = 0; i < row; i++)
    {
        delete[] arr2D[i]; // delete row
    }
    delete[] arr2D; // delete row pointer array
    delete[] arr1D;
    return 0;
}