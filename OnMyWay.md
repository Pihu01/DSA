1. Two pointers Pattern:(04 Sept 2025)
   
  When to use: When you need to find pairs, compare elements, or work with sorted arrays.
  Key Idea: Use two pointers moving towards each other or in the same direction.
  
  - https://www.youtube.com/watch?v=QzZ7nmouLTI

 -  https://bytebytego.com/courses/coding-patterns/two-pointers/introduction-to-two-pointers?fpr=javarevisited

 -  https://bytebytego.com/exercises/coding-patterns
   
 - https://leetcode.com/discuss/post/1688903/solved-all-two-pointers-problems-in-100-z56cn/


 - https://github.com/Chanda-Abdul/Several-Coding-Patterns-for-Solving-Data-Structures-and-Algorithms-Problems-during-Interviews/blob/main/%E2%9C%85%20%20Pattern%2002%3A%20Two%20Pointers.md




##### MethodO1: 2 pair - Two pointer

               
      def pairsum(target):
          print("target",target)
          a=[1,2,3,4,5]
          
          for i in range(len(a)-1):
              for j in range(i+1, len(a)-1):
                  if(a[i]+a[j] == target):
                    return i,j
          return False
  
      print(pairsum(7))



