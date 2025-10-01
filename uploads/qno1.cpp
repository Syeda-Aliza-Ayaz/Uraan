#include<iostream>
using namespace std;
struct Node{//node structure
    int data;
    Node* next;
    Node(int val){//constructor to inilialize node
        data=val;//set data to val
        next=NULL;//set next to NULL
    }
};
Node* mergeTwoLists(Node* l1, Node* l2){//function to merge two sorted linked lists
if(!l1) return l2;//if list1 is empty, return list2
if(!l2) return l1;//if list2 is empty, return list1
Node* dummy=new Node(0);//dummy node to help with merging
Node* tail=dummy;//tail pointer to build the new list
while(l1!=NULL && l2!=NULL){//traverse both lists
    if(l1->data<=l2->data){//if current node of list1 is smaller or equal
        tail->next=l1;//link it to merged list
        l1=l1->next;//move to next node in list1
    }
    else{ //if current node of list2 is smaller
        tail->next=l2; //link it to merged list
        l2=l2->next; //move to next node in list2
    }
    tail=tail->next; //move tail to the last node in merged list
   
}
 if(l1!=NULL) tail->next=l1;//if any nodes left in list1, link them
 if(l2!=NULL) tail->next=l2;//if any nodes left in list2, link them
Node* result=dummy->next; //head of merged list is next of dummy
delete dummy; //free the dummy node to avoid memory leak
return result; //return head of merged list 
}

void display(Node* head){//function to display the LL
    Node* temp = head;
    while(temp != NULL){//traverse until end of list
        cout << temp->data << " ";
        temp = temp->next;
    }
    cout << "NULL" << endl;
}
void insertAtEnd(Node*& head, int val) {//function to insert a new node at the end of the list
    Node* newNode = new Node(val);
    if (!head) {//if list is empty, new node becomes the head
        head = newNode;
        return;
    }
    Node* temp = head;//if not, temporary pointer to traverse the list
    while (temp->next != NULL) { //traverse to the last node
        temp = temp->next;
    }
    temp->next = newNode; //link last node to new node
} 
void freeList(Node* &head) { //function to free the allocated memory of the list
    while (head != NULL) {
        Node* temp = head;
        head = head->next;
        delete temp;
    }
}

int main(){
    //taking input for two sorted linked lists
    Node* list1=NULL; 
    Node* list2=NULL;
    int n1,n2,val;
    cout<<"Enter number of elements in first LL: ";
    cin>>n1;
    cout<<"Enter elements in sorted order: ";
    for(int i=0;i<n1;i++){
        cin>>val;
        insertAtEnd(list1,val);
    }
    cout<<"Enter number of elements in second LL: ";
    cin>>n2;
    cout<<"Enter elements in sorted order: ";
    for(int i=0;i<n2;i++){
        cin>>val;
        insertAtEnd(list2,val);
    }
    Node* mergedList=mergeTwoLists(list1,list2);
    cout<<"Merged sorted linked list: ";
    display(mergedList);
    freeList(mergedList);
    return 0;
}
