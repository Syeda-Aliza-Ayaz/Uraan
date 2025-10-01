public class ConstantTime {
    public static int getFirst(int[] arr) {
        return arr[0]; // Always one step → O(1)
    }

    public static void main(String[] args) {
        int[] arr = {10, 20, 30};
        System.out.println(getFirst(arr));
    }
}
