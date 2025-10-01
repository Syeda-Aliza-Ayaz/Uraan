#include <iostream>
using namespace std;
// Node class represents each element in the queue
class Node
{
public:
    int data; // Data stored in the node
    Node *next;  // Pointer to the next node
    Node *prev; //pointer to previous node
    Node(int data) // Constructor initializes node with given data
    {
        this->data = data;
        this->next = NULL;
        this->prev=NULL;
    }
};
//LinkedList class
class LinkedList{
public:
    Node *head; //pointer to head
    Node *tail; //pointer to tail
    //constructor to initialize empty list
LinkedList():head(NULL),tail(NULL){}
//insert values aat end
void insertEnd(int val){
    Node *temp=new Node(val); //create new node
    if(!head){ //check if list is empty
        head=temp; //head and tail both points to first node
        tail=temp;
        return;
    }
    //link new node after current tail
    tail->next=temp;
    temp->prev=tail;
    tail=temp; //update tail pointer
}
 // Concatenate another list (obj) to the current list
void concatenate(LinkedList &obj){
    if(!obj.head){ // check if empty
        head=obj.head;
        tail=obj.tail;
        return;
    }
    // Link tail of current list to head of obj
    tail->next=obj.head;
    obj.head->prev=tail;
    tail=obj.tail;  // Update tail pointer
}
//descending function
void Descending(){
    if(!head) return; //check if empty
    bool swap; //to check swap 
    //loop untill no swaps are made
    do{
        swap=false;
        Node* curr=head; //start from head
        //traverse till second last node
        while(curr->next){
            //compare current node to next node
            if(curr->data<curr->next->data){
                //swap data value (not nodes)
                int temp=curr->data;
                curr->data=curr->next->data;
                curr->next->data=temp;
                swap=true; //mark that swap has occurred 
            }
            curr=curr->next; //move to next
        }
    }while(swap); //repeat till no swaps to make
}
//display elements
 void Display()
    {
        if (!head)
        {
            cout << "Queue is empty" << endl;
            return;
        }
        Node *temp = head;
        cout << "Queue Elements: ";
        //traverse till end
        while(temp){
            cout << temp->data << " ";
            temp = temp->next;

        } 
        cout << endl;
    }
};
int main() {
   LinkedList L,M,N; //3 objects of class
    //insert even numbers for list L
    for(int i=2;i<=10;i+=2){
        L.insertEnd(i);
    }
    //insert odd numbers for list M
     for(int i=1;i<=9;i+=2){
        M.insertEnd(i);
    }
    //display both lists
    cout<<"List L (even): ";
    L.Display();
     cout<<"List M (odd): ";
    M.Display();
    N=L; //copy list L to list N
    N.concatenate(M); //concatenate M into N (L+M)
     cout<<"List N after concatenation of L and M: ";
    N.Display(); //display answer
    N.Descending(); //descending function call
    cout<<"List N after sorting in descending order";
    N.Display(); //after descending display
    return 0;
}