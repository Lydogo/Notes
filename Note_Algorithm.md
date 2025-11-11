# 复杂度

## 1.时间复杂度

### 定义：对于程序大致运行时间的描述

if语句:O(1)

for循环:n次单独的操作 O(n)

两层循环: O(n^2)

1e8:大约1s

## 2.空间复杂度

### 定义：对于程序大致运行空间的描述

一个int型变量所占空间大小为4byte=4b
题目中经常出现的空间限制 128mb，256mb
1mb = 1024kb
1kb = 1024b
1mb = 1024*1024b = 1e6b

1mb可以有 2.5*10^5个int变量

# 数组的基本原理
## 静态数组
### 定义
一块连续的内存空间，我们可以通过索引来访问这块内存空间中的元素，这才是数组的原始形态。
### 增
#### 情况一，数组末尾追加（append）元素
直接对索引赋值
#### 情况二，数组中间插入（insert）元素
涉及「数据搬移」，给新元素腾出空位，然后再才能插入新元素
```cpp
// 大小为 10 的数组已经装了 4 个元素
int arr[10];
for (int i = 0; i < 4; i++) {
    arr[i] = i;
}

// 在索引 2 置插入元素 666
// 需要把索引 2 以及之后的元素都往后移动一位
// 注意要倒着遍历数组中已有元素避免覆盖，不懂的话请看下方可视化面板
for (int i = 4; i > 2; i--) {
    arr[i] = arr[i - 1];
}

// 现在第 3 个位置空出来了，可以插入新元素
arr[2] = 666;
```

## 动态数组
编程语言为了方便我们使用，在静态数组的基础上帮我们添加了一些常用的 API，比如 push, insert, remove 等等方法，这些 API 可以让我们更方便地操作数组元素，不用自己去写代码实现这些操作
数组随机访问的超能力源于数组连续的内存空间，而连续的内存空间就不可避免地面对数据搬移和扩缩容的问题。
```cpp
// 创建动态数组
// 不用显式指定数组大小，它会根据实际存储的元素数量自动扩缩容
vector<int> arr;

for (int i = 0; i < 10; i++) {
    // 在末尾追加元素，时间复杂度 O(1)
    arr.push_back(i);
}

// 在中间插入元素，时间复杂度 O(N)
// 在索引 2 的位置插入元素 666
arr.insert(arr.begin() + 2, 666);

// 在头部插入元素，时间复杂度 O(N)
arr.insert(arr.begin(), -1);

// 删除末尾元素，时间复杂度 O(1)
arr.pop_back();

// 删除中间元素，时间复杂度 O(N)
// 删除索引 2 的元素
arr.erase(arr.begin() + 2);

// 根据索引查询元素，时间复杂度 O(1)
int a = arr[0];

// 根据索引修改元素，时间复杂度 O(1)
arr[0] = 100;

// 根据元素值查找索引，时间复杂度 O(N)
int index = find(arr.begin(), arr.end(), 666) - arr.begin();
```

## 数组中的双指针使用
### 原地修改
让慢指针 slow 走在后面，快指针 fast 走在前面探路，找到一个不重复的元素就赋值给 slow 并让 slow 前进一步。
这样，就保证了 nums[0..slow] 都是无重复的元素，当 fast 指针遍历完整个数组 nums 后，nums[0..slow] 就是整个数组去重之后的结果。

### 二分查找

### 反转数组
```cpp
void reverseString(vector<char>& s) {
    // 一左一右两个指针相向而行
    int left = 0, right = s.size() - 1;
    while (left < right) {
        // 交换 s[left] 和 s[right]
        char temp = s[left];
        s[left] = s[right];
        s[right] = temp;
        left++;
        right--;
    }
}
```

### 回文串判断
```cpp
bool isPalindrome(string s) {
    // 一左一右两个指针相向而行
    int left = 0, right = s.length() - 1;
    while (left < right) {
        if (s[left] != s[right]) {
            return false;
        }
        left++;
        right--;
    }
    return true;
}
```











# 链表
一条链表并不需要一整块连续的内存空间存储元素。链表的元素可以分散在内存空间的天涯海角，通过每个节点上的 next, prev 指针，将零散的内存块串联起来形成一个链式结构。
优点：提高内存的利用效率；不需要考虑扩缩容和数据搬移的问题
缺点：不支持通过索引快速访问元素（如果你想要访问第 3 个链表元素，你就只能从头结点开始往顺着 next 指针往后找，直到找到第 3 个节点才行。）

## 虚拟头结点
```cpp
ListNode dummy(-1),*p = &dummy;


//存放小于元素的链表的虚拟头结点
ListNode* dummy1 = new ListNode(-1);
//存放大于元素的链表的虚拟头结点
ListNode* dummy2 = new ListNode(-1);
// p1, p2 指针负责生成结果链表
ListNode* p1 = dummy1, *p2 = dummy2;
```

## 单链表的倒数第 k 个节点
只遍历一次链表，就算出倒数第 k 个节点:
双指针
1）先让一个指针 p1 指向链表的头节点 head，然后走 k 步
2）现在的 p1，只要再走 n - k 步，就能走到链表末尾的空指针
3）再用一个指针 p2 指向链表头节点 head
4）让 p1 和 p2 同时向前走
此时p1 走到链表末尾的空指针时前进了 n - k 步，p2 也从 head 开始前进了 n - k 步，停留在第 n - k + 1 个节点上，即恰好停链表的倒数第 k 个节点上
```cpp
// 返回链表的倒数第 k 个节点
ListNode* findFromEnd(ListNode* head, int k) {
    ListNode* p1 = head;
    // p1 先走 k 步
    for (int i = 0; i < k; i++) {
        p1 = p1 -> next;
    }
    ListNode* p2 = head;
    // p1 和 p2 同时走 n - k 步
    while (p1 != nullptr) {
        p2 = p2 -> next;
        p1 = p1 -> next;
    }
    // p2 现在指向第 n - k + 1 个节点，即倒数第 k 个节点
    return p2;
}
```

## 单链表的中点
快慢指针
1）让两个指针 slow 和 fast 分别指向链表头结点 head
每当慢指针 slow 前进一步，快指针 fast 就前进两步，这样，当 fast 走到链表末尾时，slow 就指向了链表中点。
```cpp
class Solution {
public:
    ListNode* middleNode(ListNode* head) {
        // 快慢指针初始化指向 head
        ListNode* slow = head;
        ListNode* fast = head;
        // 快指针走到末尾时停止
        while (fast != nullptr && fast->next != nullptr) {
            // 慢指针走一步，快指针走两步
            slow = slow->next;
            fast = fast->next->next;
        }
        // 慢指针指向中点
        return slow;
    }
};
```

## 判断链表是否包含环
快慢指针
每当慢指针 slow 前进一步，快指针 fast 就前进两步。
如果 fast 最终能正常走到链表末尾，说明链表中没有环；如果 fast 走着走着竟然和 slow 相遇了，那肯定是 fast 在链表中转圈了，说明链表中含有环。
```cpp
class Solution {
public:
    bool hasCycle(ListNode *head) {
        // 快慢指针初始化指向 head
        ListNode *slow = head, *fast = head;
        // 快指针走到末尾时停止
        while (fast != nullptr && fast->next != nullptr) {
            // 慢指针走一步，快指针走两步
            slow = slow->next;
            fast = fast->next->next;
            // 快慢指针相遇，说明含有环
            if (slow == fast) {
                return true;
            }
        }
        // 不包含环
        return false;
    }
};
```

### 如何寻找环起点
当快慢指针相遇时，让其中任一个指针指向头节点，然后让它俩以相同速度前进，再次相遇时所在的节点位置就是环开始的位置。
我们假设快慢指针相遇时，慢指针 slow 走了 k 步，那么快指针 fast 一定走了 2k 步
fast 一定比 slow 多走了 k 步，这多走的 k 步其实就是 fast 指针在环里转圈圈，所以 k 的值就是环长度的「整数倍」
假设相遇点距环的起点的距离为 m，那么结合上图的 slow 指针，环的起点距头结点 head 的距离为 k - m，也就是说如果从 head 前进 k - m 步就能到达环起点。
巧的是，如果从相遇点继续前进 k - m 步，也恰好到达环起点。因为结合上图的 fast 指针，从相遇点开始走k步可以转回到相遇点，那走 k - m 步肯定就走到环起点了
所以，只要我们把快慢指针中的任一个重新指向 head，然后两个指针同速前进，k - m 步后一定会相遇，相遇之处就是环的起点了

## 两个链表是否相交
如果用两个指针 p1 和 p2 分别在两条链表上前进，并不能同时走到公共节点，也就无法得到相交节点 c1。
解决这个问题的关键是，通过某些方式，让 p1 和 p2 能够同时到达相交节点 c1。
所以，我们可以让 p1 遍历完链表 A 之后开始遍历链表 B，让 p2 遍历完链表 B 之后开始遍历链表 A，这样相当于「逻辑上」两条链表接在了一起。
如果这样进行拼接，就可以让 p1 和 p2 同时进入公共部分，也就是同时到达相交节点 c1

```cpp
class Solution {
public:
    ListNode* getIntersectionNode(ListNode* headA, ListNode* headB) {
        // p1 指向 A 链表头结点，p2 指向 B 链表头结点
        ListNode* p1 = headA;
        ListNode* p2 = headB;
        while (p1 != p2) {
            // p1 走一步，如果走到 A 链表末尾，转到 B 链表
            p1 = (p1 == nullptr) ? headB : p1->next;
            // p2 走一步，如果走到 B 链表末尾，转到 A 链表
            p2 = (p2 == nullptr) ? headA : p2->next;
        }
        return p1;
    }
};
```











# 队列和栈
队列只能在一端插入元素，另一端删除元素；栈只能在某一端插入和删除元素





# 哈希表
哈希表和我们常说的 Map（键值映射）不同，哈希表可以理解为一个加强版的数组，数组可以通过索引在 O(1) 的时间复杂度内查找到对应元素，索引是一个非负整数。

## 重点概念
### 1.key 是唯一的，value 可以重复，并且一定要用不可变类型作为哈希表的 key
因为哈希表的主要操作都依赖于哈希函数计算出来的索引，如果 key 的哈希值会变化，会导致键值对意外丢失，产生严重的 bug。
### 2.哈希表的增删查改效率不一定是 O(1) 
只有哈希函数的复杂度是 O(1)，且合理解决哈希冲突的问题，才能保证增删查改的复杂度是 O(1)。
### 3.哈希表的遍历顺序为什么会变化
因为哈希表在达到负载因子时会扩容，这个扩容过程会导致哈希表底层的数组容量变化，哈希函数计算出来的索引也会变化，所以哈希表的遍历顺序也会变化
### 4.为什么我们常说，哈希表的增删查改效率都是 O(1)
因为哈希表底层就是操作一个数组，其主要的时间复杂度来自于哈希函数计算索引和哈希冲突。只要保证哈希函数的复杂度在 O(1)，且合理解决哈希冲突的问题，那么增删查改的复杂度就都是 O(1)


## （1）导入头文件
```C++
#include<unordered_map>
```
## （2）哈希表的声明和初始化
```C++
//声明一个没有任何元素的哈希表
unordered_map<elemType_1, elemType_2> var_name; 

//其中elemType_1和elemType_2是模板允许定义的类型，如要定义一个键值对都为Int的哈希表：
unordered_map<int, int> map;

//当我们想向哈希表中添加元素时也可以直接通过下标运算符添加元素，格式为: mapName[key]=value;
//如：hmap[4] = 14;
//但是这样的添加元素的方式会产生覆盖的问题，也就是当hmap中key为4的存储位置有值时，
//再用hmap[4]=value添加元素，会将原哈希表中key为4存储的元素覆盖
hmap[4] = 14;
hmap[5] = 15;
cout << hmap[4];  //结果为15

//通过insert()函数来添加元素的结果和通过下标来添加元素的结果一样，不同的是insert()可以避免覆盖问题，
//insert()函数在同一个key中插入两次，第二次插入会失败
hmap.insert({ 5,15 });
hmap.insert({ 5,16 });
cout << hmap[5];  //结果为15

```
## （3）STL中哈希表的常用函数
```C++
//begin( )函数：该函数返回一个指向哈希表开始位置的迭代器
unordered_map<int, int>::iterator iter = hmap.begin(); //申请迭代器，并初始化为哈希表的起始位置
cout << iter->first << ":" << iter->second;

//end( )函数：作用和begin函数相同，返回一个指向哈希表结尾位置的下一个元素的迭代器
unordered_map<int, int>::iterator iter = hmap.end();

//cbegin() 和 cend()：这两个函数的功能和begin()与end()的功能相同，唯一的区别是cbegin()和cend()是面向不可变的哈希表
const unordered_map<int, int> hmap{ {1,10},{2,12},{3,13} };
unordered_map<int, int>::const_iterator iter_b = hmap.cbegin(); //注意这里的迭代器也要是不可变的const_iterator迭代器
unordered_map<int, int>::const_iterator iter_e = hmap.cend();

//(4) empty()函数：判断哈希表是否为空，空则返回true，非空返回false
bool isEmpty = hmap.empty();

//(5) size()函数：返回哈希表的大小
int size = hmap.size();

//(6) erase()函数： 删除某个位置的元素，或者删除某个位置开始到某个位置结束这一范围内的元素， 或者传入key值删除键值对
unordered_map<int, int> hmap{ {1,10},{2,12},{3,13} };
unordered_map<int, int>::iterator iter_begin = hmap.begin();
unordered_map<int, int>::iterator iter_end = hmap.end();
hmap.erase(iter_begin);  //删除开始位置的元素
hmap.erase(iter_begin, iter_end); //删除开始位置和结束位置之间的元素
hmap.erase(3); //删除key==3的键值对

//(7) at()函数：根据key查找哈希表中的元素
unordered_map<int, int> hmap{ {1,10},{2,12},{3,13} };
int elem = hmap.at(3);

//(8) clear()函数：清空哈希表中的元素
hmap.clear()

//(9) find()函数：以key作为参数寻找哈希表中的元素，如果哈希表中存在该key值则返回该位置上的迭代器，否则返回哈希表最后一个元素下一位置上的迭代器
unordered_map<int, int> hmap{ {1,10},{2,12},{3,13} };
unordered_map<int, int>::iterator iter;
iter = hmap.find(2); //返回key==2的迭代器，可以通过iter->second访问该key对应的元素
if(iter != hmap.end())  cout << iter->second;

//(10) bucket()函数：以key寻找哈希表中该元素的储存的bucket编号（unordered_map的源码是基于拉链式的哈希表，所以是通过一个个bucket存储元素）
int pos = hmap.bucket(key);

//(11) bucket_count()函数：该函数返回哈希表中存在的存储桶总数（一个存储桶可以用来存放多个元素，也可以不存放元素，并且bucket的个数大于等于元素个数）
int count = hmap.bucket_count();

//(12) count()函数： 统计某个key值对应的元素个数， 因为unordered_map不允许重复元素，所以返回值为0或1
int count = hmap.count(key);

//(13) 哈希表的遍历: 通过迭代器遍历
unordered_map<int, int> hmap{ {1,10},{2,12},{3,13} };
unordered_map<int, int>::iterator iter = hmap.begin();
for( ; iter != hmap.end(); iter++){
 cout << "key: " <<  iter->first  << "value: " <<  iter->second <<endl;
}

```

# 二叉树
## 满二叉树
每一层节点都是满的，整棵树像一个正三角形
## 完全二叉树
二叉树的每一层的节点都紧凑靠左排列，且除了最后一层，其他每层都必须是满的
**完全二叉树的左右子树中，至少有一棵是满二叉树
## 二叉搜索树 BST
对于树中的每个节点，其左子树的每个节点的值都要小于这个节点的值，右子树的每个节点的值都要大于这个节点的值。可以简单记为「左小右大」。

## 高度平衡二叉树
「每个节点」的左右子树的高度差不超过 1。
假设高度平衡二叉树中共有N个节点，那么高度平衡二叉树的高度是O(logN)

## 自平衡二叉树
在增删二叉树节点时对树的结构进行一些调整，那么就可以让树的高度始终是平衡的，这就是自平衡二叉树（Self-Balanced Binary Tree）
自平衡的二叉树有很多种实现方式，最经典的就是 红黑树，一种自平衡的二叉搜索树

**二叉树只有递归遍历和层序遍历这两种，再无其他。递归遍历可以衍生出 DFS 算法，层序遍历可以衍生出 BFS 算法

## 二叉树的前前/中/后序三种遍历

## 最常用的层序遍历写法
```cpp
void levelOrderTraverse(TreeNode* root) {
    if (root == nullptr) {
        return;
    }
    queue<TreeNode*> q;
    q.push(root);
    // 记录当前遍历到的层数（根节点视为第 1 层）
    int depth = 1;

    while (!q.empty()) {
        int sz = q.size();
        for (int i = 0; i < sz; i++) {
            TreeNode* cur = q.front();
            q.pop();
            // 访问 cur 节点，同时知道它所在的层数
            cout << "depth = " << depth << ", val = " << cur->val << endl;

            // 把 cur 的左右子节点加入队列
            if (cur->left != nullptr) {
                q.push(cur->left);
            }
            if (cur->right != nullptr) {
                q.push(cur->right);
            }
        }
        depth++;
    }
}
```


# 排序稳定性
对于序列中的相同元素，如果排序之后它们的相对位置没有发生改变，则称该排序算法为「稳定排序」，反之则为「不稳定排序」
















# 二分查找

## 问题引入：返回有序数组中第一个>=8的数的位置

### 高效做法：L和R分别指向询问的左右边界

### 写法（左开右开）

```C++
class Solution {
    // lower_bound 返回最小的满足 nums[i] >= target 的下标 i
    // 如果数组为空，或者所有数都 < target，则返回 nums.size()
    // 要求 nums 是非递减的，即 nums[i] <= nums[i + 1]
    int lower_bound(vector<int>& nums, int target) {
        int left = -1, right = nums.size(); // 开区间 (left, right)
        while (left + 1 < right) { // 区间不为空
            // 循环不变量：
            // nums[left] < target
            // nums[right] >= target
            int mid = left + (right - left) / 2; //防止溢出
            if (nums[mid] >= target) {
                right = mid; // 范围缩小到 (left, mid)
            } 
            else {
                left = mid; // 范围缩小到 (mid, right)
            }
        }
        // 循环结束后 left+1 = right
        // 此时 nums[left] < target 而 nums[right] >= target
        // 所以 right 就是第一个 >= target 的元素下标
        return right;
    }
};
```

### 四种情况
上题是>=,而对于>,<=,<又应该怎么操作？

对于整数数组来说：
1）>target 等价于 ≥ target+1
2）<=target 等价于 (>target)的左边那个数
3）<target 等价于 (≥ target)的左边那个数




# 滑动窗口算法 Sliding Window
通过在数据结构（如数组、字符串）上维护一个窗口，来减少不必要的重复计算，从而提高效率。窗口可以是固定大小的，也可以是可变大小的。滑动窗口通常有两个指针（或索引）来表示窗口的左右边界，这两个指针会随着算法的进行在数据结构上滑动，计算结果。

主要用来解决子数组问题，比如让你寻找符合某个条件的最长/最短子数组。

```C++
// 滑动窗口算法伪码框架
void slidingWindow(string s) {
    // 用合适的数据结构记录窗口中的数据，根据具体场景变通
    // 比如说，我想记录窗口中元素出现的次数，就用 map
    // 如果我想记录窗口中的元素和，就可以只用一个 int
    auto window = ...

    int left = 0, right = 0;
    while (right < s.size()) {
        // c 是将移入窗口的字符
        char c = s[right];
        window.add(c);
        // 增大窗口
        right++;

        // 进行窗口内数据的一系列更新
        ...

        // *** debug 输出的位置 ***
        printf("window: [%d, %d)\n", left, right);
        // 注意在最终的解法代码中不要 print
        // 因为 IO 操作很耗时，可能导致超时

        // 判断左侧窗口是否要收缩
        while (window needs shrink) {
            // d 是将移出窗口的字符
            char d = s[left];
            window.remove(d);
            // 缩小窗口
            left++;

            // 进行窗口内数据的一系列更新
            ...
        }
    }
}
```
基于这个框架，遇到子串/子数组相关的题目，你只需要回答以下三个问题：
1、什么时候应该移动 right 扩大窗口？窗口加入字符时，应该更新哪些数据？
2、什么时候窗口应该暂停扩大，开始移动 left 缩小窗口？从窗口移出字符时，应该更新哪些数据？
3、什么时候应该更新结果？




# 动态规划
## 动态规划三要素
动态规划问题的一般形式就是求最值
求解动态规划的核心问题是穷举
1.列出正确的「状态转移方程」
2.问题是否具备「最优子结构」
3.存在「重叠子问题」，需要使用「备忘录」或者「DP table」来优化穷举过程

### 动态规划代码框架
```cpp
# 自顶向下递归的动态规划
def dp(状态1, 状态2, ...):
    for 选择 in 所有可能的选择:
        # 此时的状态已经因为做了选择而改变
        result = 求最值(result, dp(状态1, 状态2, ...))
    return result

# 自底向上迭代的动态规划
# 初始化 base case
dp[0][0][...] = base case
# 进行状态转移
for 状态1 in 状态1的所有取值：
    for 状态2 in 状态2的所有取值：
        for ...
            dp[状态1][状态2][...] = 求最值(选择1，选择2...)
```

### 动态规划解法的两种表现形式：
#### 1、带备忘录的递归解法「自顶向下」
一个递归函数带一个 memo 备忘录。
```cpp
int fib(int n) {
    // 备忘录全初始化为 -1
    // 因为斐波那契数肯定是非负整数，所以初始化为特殊值 -1 表示未计算

    // 因为数组的索引从 0 开始，所以需要 n + 1 个空间
    // 这样才能把 `f(0) ~ f(n)` 都记录到 memo 中
    vector<int> memo(n + 1, -1);

    // 进行带备忘录的递归
    return dp(memo, n);
}

// 带着备忘录进行递归
int dp(vector<int>& memo, int n) {
    // base case
    if (n == 0 || n == 1) {
        return n;
    }
    // 已经计算过，不用再计算了
    if (memo[n] != -1) {
        return memo[n];
    }
    // 在返回结果之前，存入备忘录
    memo[n] = dp(memo, n - 1) + dp(memo, n - 2);
    return memo[n];
}
```
#### 2、DP table 的迭代解法「自底向上」
用 for 循环去迭代 dp 数组进行求解。
```cpp
int fib(int n) {
    if (n == 0 || n == 1) {
        return n;
    }
    // dp table
    vector<int> dp(n + 1);
    // base case
    dp[0] = 0; dp[1] = 1;
    // 状态转移
    for (int i = 2; i <= n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }

    return dp[n];
}
```
# 回溯算法

