#include<iostream>

using namespace std;

// Maximum size of the stack
const int MAX_SIZE = 10;

// The Stack class using a fixed-size array
class Stack {
private:
    int arr[MAX_SIZE];
    int top; // Index of the top element
    
public:
    // Constructor
    Stack() {
        top = -1; // Initialize top to -1 to indicate an empty stack
    }
    
    // Pushes an element onto the stack
    void push(int data) {
        if (isFull()) {
            cout << "Stack overflow! Cannot push " << data << "." << endl;
            return;
        }
        top++;
        arr[top] = data;
        cout << "Pushed " << data << " to the stack." << endl;
    }
    
    // Pops an element from the stack
    void pop() {
        if (isEmpty()) {
            cout << "Stack underflow! Cannot pop." << endl;
            return;
        }
        cout << "Popped " << arr[top] << " from the stack." << endl;
        top--;
    }
    
    // Returns the top element without removing it
    int peek() {
        if (isEmpty()) {
            cout << "Stack is empty." << endl;
            return -1; // Or handle with an exception
        }
        return arr[top];
    }
    
    // Checks if the stack is empty
    bool isEmpty() {
        return top == -1;
    }
    
    // Checks if the stack is full
    bool isFull() {
        return top == MAX_SIZE - 1;
    }
};

int main() {
    Stack s;
    
    s.push(10);
    s.push(20);
    s.push(30);
    
    cout << "Top element is: " << s.peek() << endl;
    
    s.pop();
    s.pop();
    
    cout << "Top element is: " << s.peek() << endl;
    
    s.pop();
    s.pop(); // This will show the "Stack underflow" message
    
    return 0;
}