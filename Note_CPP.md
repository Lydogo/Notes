# CPP Learning Note
## 1.名字空间namespace
为防止名字冲突(出现同名),C++引入了名字空间(namespace)，通过::运算符限定某个名字属于哪个名字空间

*如  “计算机1702”::“李平”*

*如  “信计1603”::“李平”*

```C++
#include <iostream>
#include <cstdio>
 namespace first
 {
    int a;
    void f(){/*...*/}
    int g(){/*...*/}
 }
 namespace second
 {
    double a;
    double f(){/*...*/}
    char g;
 }
 int main ()
 {
    first::a = 2;
    second::a = 6.453;
    first::a = first::g()+second::f();
    second::a = first::g()+6.453;
    printf("%d\n",first::a);
    printf("%lf\n",second::a);
    return 0;
 }
```
### 声明的三种方法：
1.using namespace X; //引入整个名字空间，后续就不需要再加限定词了
2.using X::name ; //使用单个名字，后续yongname不用限定词
3.X::name; //程序中加上名字空间前缀，如X::

## 2.C++的输入输出流库(头文件iostream)
将输入输出看成一个流，对数据(变量和常量进行输入输出)
输出运算符: << 
输入运算符: >> 
其中有cout和cin分别代表标准输出流对象(屏幕窗口)和标准输入流对象(键盘);标准库中的名字都属于标准名字空间std

```C++
#include <iostream>
#include <cmath>
using std::cout; //使用单个名字
int main()
{
   double a;
   cout << "从键盘输入一个数" << std::endl; //endl表示换行符，并强制输出
   std::cin >> a; //  通过“名字限定”std::cin,
                  //cin是代表键盘的输入流对象，>>等待键盘输入一个实数
   a = sin(a);
   cout << a;    //cout是代表屏幕窗口的输出流对象
   return 0;
}

#include <iostream>  //标准输入输出头文件
#include <cmath>
using namespace std; //引入整个名字空间std中的所有名字
                        //cout cin都属于名字空间std;
int main() 
{
   double a;
   cout << "从键盘输入一个数" << endl;
   cin >> a;
   a = sin(a);
   cout << a;
   return 0;
}
```

## 3.变量“即用即定义”，且可用表达式初始化

```C++
#include <iostream>
using namespace std;
int main ()
{
   double a = 12 * 3.25;
   double b = a + 1.112;
   cout << "a contains : " << a << endl;
   cout << "b contains: " << b << endl;
   a = a * 2 + b;
   double c = a + b * a; //“即用即定义”，且可用表达式初始化
   cout << "c contains: " << c << endl;
 }
```

## 4.程序块{}内部作用域
内部作用域可定义域外部作用域同名的变量，在该块里就隐藏了外部变量
即：一个变量的定义只存在于其定义域内

```C++
#include <iostream>
using namespace std;
int main ()
{
   double a;
   cout << "Type a number: ";
   cin >> a;
   {
      int a = 1; // "int a"隐藏了外部作用域的“double a"   
      a = a * 10 + 4;
      cout << "Local number: " << a << endl;
   }
   cout << "You typed: " << a << endl; //main作用域的“double a"
   return 0;
}
```

## 5.for循环语句可以定义局部变量

```C++
#include <iostream>
using namespace std;
int main ()
{
   int i = 0;
   for (int i = 0; i < 4; i++)  //
   {
      cout << i << endl;
   }
   cout << "i contains: " << i << endl; //这里仍然输出0

   for (i = 0; i < 4; i++)
   {
      for (int i = 0; i < 4; i++)
      {                                  
         cout << i<< " ";
      }
      cout << endl;
   }
   return 0;
}
```

## 6.访问和内部作用域变量同名的全局变量，要用全局作用域限定 ::


```C++
#include <iostream>
using namespace std;
double a = 128;
int main (){
   double a = 256;
   cout << "Local a: " << a << endl;
   cout << "Global a: " <<::a << endl; //::是全局作用域限定
   return 0;
}
```

## 7.C++引入了“引用类型”，即一个变量是另一个变量的别名


```C++
#include <iostream>
using namespace std;
int main()
{
    double a = 3.1415927;
    double &b = a; // b 是 a 的别名，b 就是 a

    b = 89; // 修改 b 的值，也就是修改 a 的值
    cout << "a contains: " << a << endl; // 输出 89
    return 0;
}
```
### a 和 b 交换失败，原因：
#### 1)按值传递：在 swap 函数中，x 和 y 是 a 和 b 的副本。x 和 y 的修改不会影响 a 和 b 的值。
#### 2)局部变量：x 和 y 是 swap 函数的局部变量，它们的生命周期仅限于 swap 函数的执行期间。当 swap 函数执行完毕后，x 和 y 就会被销毁。
```C++
#include <iostream>
using namespace std;

void swap(int x, int y) {
    cout << "swap 函数内交换前：" << x << " " << y << endl;
    int t = x;
    x = y;
    y = t;
    cout << "swap 函数内交换后：" << x << " " << y << endl;
}

int main() {
    int a = 3, b = 4;
    swap(a, b);
    cout << a << ", " << b << endl; // 这里 a 和 b 的值没有交换
    return 0;
}
```
### 用指针和取地址符号实现交换函数:
```C++
#include <iostream>
using namespace std;

void swap(int *x, int *y)
{
    cout << "swap 函数内交换前：" << *x << " " << *y << endl;
    int t = *x;
    *x = *y;
    *y = t;
    cout << "swap 函数内交换后：" << *x << " " << *y << endl;
}

int main()
{
    int a = 3, b = 4;
    swap(&a, &b); // &a 赋值给 x，&b 赋值给 y
                  // x 和 y 分别是 int* 指针，指向 a 和 b
                  // *x 和 *y 就是 a 和 b
    cout << a << ", " << b << endl; // 输出交换后的值
    return 0;
}
```
### 使用引用符号 & 实现交换函数：
代替传值可提高效率
```C++
#include <iostream>
using namespace std;

void swap(int &x, int &y)
{
    cout << "swap 函数内交换前：" << x << " " << y << endl;
    int t = x;
    x = y;
    y = t;
    cout << "swap 函数内交换后：" << x << " " << y << endl;
}

int main()
{
    int a = 3, b = 4;
    swap(a, b); // x 和 y 将分别是 a 和 b 的引用，即 x 就是 a，y 就是 b
    cout << a << ", " << b << endl; // 输出交换后的值
    return 0;
}
```

如果不希望无意中修改实参，可以用 const 修饰符：
```C++
#include <iostream>
using namespace std;

void change(double &x, const double &y, double z) {
    x = 100; // 修改 x 的值
    // y = 200; // 错误！y 不可修改，是 const double &
    z = 300; // 修改 z 的值（不影响实参，因为 z 是按值传递）
}

int main() {
    double a, b, c; // 内置类型变量未提供初始化式，默认初始化为 0
    change(a, b, c);
    cout << a << ", " << b << ", " << c << endl; // 输出修改后的值
    return 0;
}
```
## 8.inline内联函数
对于不包含循环的简单函数，建议用inline关键字声明，编译器将内联函数调用用其代码展开，称为“内联展开”，避免函数调用开销，提高程序执行效率


```C++
#include <iostream>
#include <cmath>
using namespace std;
inline double distance(double a, double b) {
return sqrt(a * a + b * b);
}
int main() 
{
double k = 6, m = 9;
// 下面2行将产生同样的代码:
cout << distance(k, m) << endl;
cout << sqrt(k * k + m * m) << endl;
return 0;
}
```

## 9.通过 try-catch处理异常情况
把正常代码放在try块，catch中捕获try块抛出的异常

### 9.1 异常处理的三个关键字
try：包裹可能抛出异常的代码块。

throw：抛出异常。可以是任何类型的值（如整数、字符串、对象等）。

catch：捕获并处理异常。必须指定捕获的异常类型。

### 9.2 异常处理的工作流程
程序执行 try 块中的代码。

如果 try 块中的代码抛出异常（通过 throw），程序立即跳转到匹配的 catch 块。

catch 块处理异常后，程序继续执行 catch 块之后的代码。

```C++
#include <iostream>
#include <cmath>
using namespace std;

int main()
{
    int a, b;

    // 输入一个数字
    cout << "Type a number: ";
    cin >> a;
    cout << endl;

    // 异常处理示例 1
    try {
        if (a > 100) throw 100;  // 如果 a > 100，抛出整数异常 100
        if (a < 10) throw 10;    // 如果 a < 10，抛出整数异常 10
        throw "hello";           // 否则，抛出字符串异常 "hello"
    }
    catch (int result) {
        // 捕获整数异常
        cout << "Result is: " << result << endl;
        b = result + 1;  // 对捕获的整数进行处理
    }
    catch (const char* s) {  // 修改为 const char*，因为字符串常量不可修改
        // 捕获字符串异常
        cout << "haha " << s << endl;
    }

    // 输出变量 b 的值
    cout << "b contains: " << b << endl;
    cout << endl;

    // 异常处理示例 2：判断数字的性质
    char zero[] = "zero";
    char pair[] = "pair";
    char notprime[] = "not prime";
    char prime[] = "prime";

    try {
        if (a == 0) throw zero;  // 如果 a 为 0，抛出异常 "zero"
        if ((a / 2) * 2 == a) throw pair;  // 如果 a 是偶数，抛出异常 "pair"

        // 判断 a 是否为质数
        for (int i = 3; i <= sqrt(a); i++) {
            if ((a / i) * i == a) throw notprime;  // 如果 a 不是质数，抛出异常 "not prime"
        }

        throw prime;  // 如果 a 是质数，抛出异常 "prime"
    }
    catch (const char* conclusion) {  // 捕获字符串异常
        cout << "异常结果是： " << conclusion << endl;
    }
    catch (...) {  // 捕获所有其他异常
        cout << "其他异常情况都在这里捕获 " << endl;
    }

    cout << endl;
    return 0;
}
```

## 10. 默认形参： 函数的形参可带有默认值。必须一律在最右边

```C++
#include <iostream>
using namespace std;
double test(double a, double b = 7) 
{
   return a - b;
}
int main()
{
cout << test(14, 5) << endl;
cout << test(14) << endl;
return 0;
}
```

## 11.函数重载
C++允许函数同名，只要它们的形参不一样(个数或对应参数类型)，调用函数时将根据实参和形参的匹配选择最佳函数，如果有多个难以区分的最佳函数，则变化一起报错！注意：不能根据返回类型区分同名函数

```C++
#include <iostream>
using namespace std;
double add(double a, double b) 
{
   return a + b;
}
int add(int a, int b)
{
   return a + b;
}

```

## 12.运算符重载
```C++
#include <iostream>
using namespace std;
struct Vector2
{
double x;
double y;
};
Vector2 operator * (double a, Vector2 b)
{
    Vector2 r;
    r.x = a * b.x;
    r.y = a * b.y;
    return r;
}
```
### 模板函数
可以对任何能比较大小（<）的类型使用该模板，让编译器自动生成一个针对该数据类型的具体函数
```C++
#include <iostream>
using namespace std;

template<class T>
T minValue(T a, T b) {
    if (a < b) return a;
    else return b;
}

int main() {
    int i = 3, j = 4;
    cout << "min of " << i << " and " << j << " is " << minValue(i, j) << endl;

    double x = 3.5, y = 10;
    cout << "min of " << x << " and " << y << " is " << minValue(x, y) << endl;

    // 但是，不同类型的怎么办？
    // 下面的代码会报错，因为模板参数类型不一致
    // cout << "min of " << i << " and " << y << " is " << minValue(i, y) << endl;

    return 0;
}

// 改进版本：支持两种不同类型的比较
// 可以对任何能比较大小（<）的类型使用该模板
// 让编译器自动生成一个针对该数据类型的具体函数
#include <iostream>
using namespace std;

template<class T1, class T2>
T1 minValue(T1 a, T2 b) {
    if (a < b) return a;
    else return (T1)b; // 强制转化为 T1 类型
}

int main()
{
    int i = 3, j = 4;
    cout << "min of " << i << " and " << j << " is " << minValue(i, j) << endl;

    double x = 3.5, y = 10;
    cout << "min of " << x << " and " << y << " is " << minValue(x, y) << endl;

    // 支持不同类型的比较
    cout << "min of " << i << " and " << y << " is " << minValue(i, y) << endl;

    return 0;
}

```

## 14.内存
### 14.1内存的基本概念
在 C++ 中，程序运行时使用的内存分为以下几个区域：
1)栈（Stack）：用于存储局部变量、函数参数等。栈内存由编译器自动分配和释放，速度快但空间有限。
2)堆（Heap）：用于动态内存分配，程序员手动管理（通过 new 和 delete）。堆空间较大，但分配和释放速度较慢。
3)全局/静态存储区：用于存储全局变量和静态变量，程序结束时释放。
4)常量存储区：用于存储常量，如字符串常量。

野指针是指指向已释放或无效内存的指针。例如：
```C++
delete dp;  // 释放内存
*dp = 10;   // 错误：dp 现在是野指针
```
解决方法：在释放内存后，将指针置为 nullptr：

```C++
delete dp;
dp = nullptr;  // 避免野指针
```

### 14.2动态内存分配
关键字 new 和 delete 比C语言的malloc/alloc/realloc和free更好，可以对类对象调用初始化构造函数或销毁析构函数

new 分配正好容纳double值的内存块（如4或8个字节）并返回这个内存块的地址，而且地址的类型是double *
这个地址被保存在dp中，dp指向这个新内存块，不再是原来d那个内存块了,但目前这个内存块的值是未知的
注意：
new 分配的是堆存储空间，即所有程序共同拥有的自由内存空间,而d,dp等局部变量是这个程序自身的静态存储空间
new会对这个double元素调用double类型的构造函数做初始化，比如初始化为0

```C++
#include <iostream>
#include <cstring>
using namespace std;

int main() {
    // 变量d是一块存放double值的内存块
    double d = 3.14;

    // 指针变量dp：保存double类型的地址的变量
    // dp的值类型是double *
    // dp是存放double *类型值的内存块
    double *dp;

    // 取地址运算符&用于获得一个变量的地址
    // 将double变量d的地址(指针)保存到double*指针变量dp中
    // dp和&d的类型都是double *
    dp = &d;

    // 解引用运算符*用于获得指针变量指向的那个变量(C++中也称为对象)
    // *dp就是dp指向的那个d
    *dp = 4.14;

    // 输出dp指向的double内存块的值
    cout << "*dp= " << *dp << " d=:" << d << endl;

    // 输入dp指向的double内存块的值
    cout << "Type a number: ";
    cin >> *dp;
    cout << "*dp= " << *dp << " d=:" << d << endl;
    dp = new double;

    // *dp指向的double内存块的值变成45.3
    *dp = 45.3;

    // 输入dp指向的double内存块的值
    cout << "Type a number: ";
    cin >> *dp;
    cout << "*dp= " << *dp << endl;  //这里d的值不变，因为现在dp指向的是一个新的堆上的内存

    // 修改dp指向的double内存块的值45.3+5
    *dp = *dp + 5;
    cout << "*dp= " << *dp << endl;

    // delete 释放dp指向的动态分配的double内存块
    delete dp;

    // new 分配了可以存放5个double值的内存块，
    // 返回这块连续内存的起始地址，而且指针类型是double *，
    // 实际是第一个double元素的地址
    // new会对每个double元素调用double类型的构造函数做初始化，比如初始化为0
    dp = new double[5];

    // dp[0]等价于 *(dp+0)即*dp，也即是第1个double元素的内存块
    dp[0] = 4456;

    // dp[1]等价于 *(dp+1)，也即是第2个double元素的内存块
    dp[1] = dp[0] + 567;

    // 输出dp数组的前两个元素
    cout << "d[0]=: " << dp[0] << " d[1]=: " << dp[1] << endl;

    // 释放dp指向的多个double元素占据的内存块，
    // 对每个double元素调用析构函数以释放资源
    // 缺少[]，只释放第一个double元素的内存块，这叫“内存泄漏”
    delete[] dp;

    // new 可以分配随机大小的double元素，
    // 而静态数组则必须是编译期固定大小，即大小为常量
    // 如 double arr[20];
    int n = 8;
    dp = new double[n];

    // 通过下标访问每个元素
    for (int i = 0; i < n; i++) {
        dp[i] = i;
    }

    // 通过指针访问每个元素
    double *p = dp;
    for (int i = 0; i < n; i++) {
        cout << *(p + i) << endl; // p[i]或dp[i]
    }
    cout << endl;

    // 通过指针遍历数组
    for (double *p = dp, *q = dp + n; p < q; p++) {
        cout << *p << endl;
    }
    cout << endl;

    // 释放dp指向的动态分配的double数组
    delete[] dp;

    // 动态分配字符数组
    char *s;
    s = new char[100];

    // 将字符串常量拷贝到s指向的字符数组内存块中
    strcpy(s, "Hello!");
    cout << s << endl;

    // 释放s指向的动态分配的字符数组
    delete[] s;

    return 0;
}
```

## 15. 类：是在C的struct类型上，增加了“成员函数”。
### 15.1 C语言中的struct
C的strcut可将一个概念或实体的所有属性组合在一起，描述同一类对象的共同属性，C++使得struct不但包含数据，还包含函数(方法)用于访问或修改类变量(对象)的这些属性。
```C++
#include <iostream>
using namespace std;
struct Date{
    int d, m, y;

    // 初始化日期
    void init(int dd, int mm, int yy) {
        d = dd;
        m = mm;
        y = yy;
    }

    // 打印日期
    void print() {
        cout << y << "-" << m << "-" << d << endl;
    }
```

### 15.2 添加天数并返回自引用 
```C++
#include <iostream>
using namespace std;
Date& add(int dd)
{
   d += dd;
   return *this;  // this是指向调用这个函数的类型对象指针，
                  // *this就是调用这个函数的那个对象。
                  // 这个成员函数返回的是“自引用”，即调用这个函数的对象本身。
                  // 通过返回自引用，可以连续调用这个函数。
                  // 例如：day.add(3).add(7);
}

    // 重载+=运算符，添加天数并返回自引用
    Date& operator+=(int dd) {
        d += dd;
        return *this; // 返回自引用，支持链式调用。
                      // 例如：(day += 5) += 7;
    }
};

int main() {
    Date day;

    // 通过类Date对象day调用类Date的print方法
    day.print();

    // 通过类Date对象day调用类Date的init方法
    day.init(4, 6, 1999);
    day.print();

    // 使用add方法添加天数
    day.add(3);
    day.add(5).add(7);
    day.print();

    // 使用重载的+=运算符添加天数
    day += 3;
    (day += 5) += 7;
    day.print();

    return 0;
}
```

## 16. 构造函数和析构函数
### 构造函数示例
```C++
#include <iostream>
using namespace std;

// Date 结构体定义
struct Date {
    int d, m, y;

    // 构造函数，初始化日期  函数名和类名一样
    Date(int dd = 1, int mm = 1, int yy = 1999) {
        d = dd;
        m = mm;
        y = yy;
        cout << "构造函数" << endl;; 



        
    }

    // 打印日期
    void print() {
        cout << y << "-" << m << "-" << d << endl;
    }

    // 析构函数
    ~Date() {
        // 目前不需要做任何释放工作，因为构造函数没申请资源
        cout << "析构函数" << endl;
    }
};

int main() {
    // 使用默认构造函数创建对象
    Date day;
    // 使用部分参数构造函数创建对象
    Date day1(2);
    // 使用部分参数构造函数创建对象
    Date day2(23, 10);
    // 使用完整参数构造函数创建对象
    Date day3(2, 3, 1999);

    // 打印各个对象的日期
    day.print();
    day1.print();
    day2.print();
    day3.print();

    return 0;
}
```

### 析构函数示例
```C++
// 
#define _CRT_SECURE_NO_WARNINGS // windows系统
#include <iostream>
#include <cstring>
using namespace std;

// Student 结构体定义
struct Student {
    char *name;
    int age;

    // 构造函数，初始化学生信息
    Student(char *n = "no name", int a = 0) {
        int len = strlen(n);
        name = new char[len+1]; // 使用 new 分配内存 需要多1个字节的空间
        strcpy(name, n);
        age = a;
        cout << "构造函数，申请了100个char元素的动态空间" << endl;
    }

    // 虚析构函数，确保正确释放资源
    virtual ~Student() {
        delete[] name; // 使用 delete[] 释放内存
        cout << "析构函数，释放了100个char元素的动态空间" << endl;
    }
};

int main() {
    cout << "Hello!" << endl << endl;

    // 使用默认构造函数创建对象
    Student a;
    cout << a.name << ", age " << a.age << endl << endl;

    // 使用部分参数构造函数创建对象
    Student b("John");
    cout << b.name << ", age " << b.age << endl << endl;

    // 修改学生年龄
    b.age = 21;
    cout << b.name << ", age " << b.age << endl << endl;

    // 使用完整参数构造函数创建对象
    Student c("Miki", 45);
    cout << c.name << ", age " << c.age << endl << endl;

    cout << "Bye!" << endl << endl;
    return 0;
}
```

## 17.访问控制、类接口
将关键字struct换成class
```C++
#include <iostream>
#include <cstring>
using namespace std;

// 学生类定义
class Student {
private:
    char *name; // 学生姓名
    int age;    // 学生年龄

public:
    // 构造函数，初始化学生信息
    Student(char *n = "no name", int a = 0) {
        name = new char[100]; // 使用 new 分配内存
        strcpy(name, n);
        age = a;
        cout << "构造函数，申请了100个char元素的动态空间" << endl;
    }

    // 虚析构函数，确保正确释放资源
    virtual ~Student() {
        delete[] name; // 使用 delete[] 释放内存
        cout << "析构函数，释放了100个char元素的动态空间" << endl;
    }

    // 获取学生姓名
    char *get_name() {
        return name;
    }

    // 获取学生年龄
    int get_age() {
        return age;
    }

    // 设置学生年龄
    void set_age(int ag) {
        age = ag;
    }
};

int main() {
    cout << "Hello!" << endl << endl;

    // 创建学生对象 a
    Student a;
    cout << a.get_name() << ", age " << a.get_age() << endl << endl;

    // 创建学生对象 b
    Student b("John");
    cout << b.get_name() << ", age " << b.get_age() << endl << endl;

    // 修改学生 b 的年龄
    b.set_age(21);
    cout << b.get_name() << ", age " << b.get_age() << endl << endl;

    cout << "Bye!" << endl << endl;
    return 0;
}
```


```C++
#include <iostream>
#include <cstring>
using namespace std;
// 数组类定义
class Array {
private:
    int size;    // 数组大小
    double *data; // 数组数据

public:
    // 构造函数，初始化数组
    Array(int s) {
        size = s;
        data = new double[s];
    }

    // 虚析构函数，确保正确释放资源
    virtual ~Array() {
        delete[] data;
    }

    // 重载 [] 运算符，访问数组元素
    double &operator[](int i) {
        if (i < 0 || i >= size) {
            cerr << endl << "Out of bounds" << endl;
            throw "Out of bounds";
        } else {
            return data[i];
        }
    }
};

int main() {
    // 创建数组对象 t
    Array t(5);

    // 访问和修改数组元素
    t[0] = 45;          // OK
    t[4] = t[0] + 6;    // OK
    cout << t[4] << endl; // OK

    // 访问越界元素，抛出异常
    try {
        t[10] = 7; // error!
    } catch (const char *msg) {
        cerr << "Caught exception: " << msg << endl;
    }

    return 0;
}
```
## 18.拷贝
拷贝： 拷贝构造函数、赋值运算符

下列赋值为什么会出错？
`“student m(s);   s = k;”`
拷贝构造函数：定义一个类对象时用同类型的另外对象初始化
赋值运算符：一个对象赋值给另外一个对象
```C++
#include <iostream>
#include <cstdlib>
using namespace std;

// 学生结构体
struct student {
    char *name; // 学生姓名
    int age;    // 学生年龄

    // 构造函数
    student(char *n = "no name", int a = 0) {
        name = new char[100]; // 动态分配100个char空间
        strcpy(name, n);     // 复制传入的姓名
        age = a;             // 设置年龄
        cout << "构造函数，申请了100个char元素的动态空间" << endl;
    }

    // 拷贝构造函数
    student(const student &s) {
        name = new char[100]; // 动态分配100个char空间
        strcpy(name, s.name); // 复制传入对象的姓名
        age = s.age;          // 复制传入对象的年龄
        cout << "拷贝构造函数，保证name指向的是自己单独的内存块" << endl;
    }

    // 拷贝赋值运算符
    student& operator=(const student &s) {
        if (this == &s) { // 自赋值检查
            return *this;
        }
        strcpy(name, s.name); // 复制传入对象的姓名
        age = s.age;          // 复制传入对象的年龄
        cout << "拷贝赋值运算符，复制name和age的值" << endl;
        return *this;         // 返回自引用
    }

    // 析构函数
    virtual ~student() {
        delete[] name; // 释放动态分配的内存
        cout << "析构函数，释放了100个char元素的动态空间" << endl;
    }
};

int main() {
    student s;               // 调用默认构造函数
    student k("John", 56);   // 调用带参构造函数
    cout << k.name << ", age " << k.age << endl;

    student m(k);            // 调用拷贝构造函数
    s = k;                   // 调用拷贝赋值运算符
    cout << s.name << ", age " << s.age << endl;

    return 0;
}
```
## 19.类体外定义方法
类体外定义方法（成员函数），必须在类定义中声明，类体外要有类作用域，否则就是全局外部函数了！
```C++
#include <iostream>
using namespace std;

// Date类用于表示日期
class Date {
    int d, m, y;  // 日、月、年

public:
    // 打印日期
    void print();

    // 构造函数，默认日期为1999年1月1日
    Date(int dd = 1, int mm = 1, int yy = 1999) {
        d = dd;
        m = mm;
        y = yy;
        cout << "构造函数" << endl;
    }

    // 虚析构函数，确保派生类对象能够正确释放资源
    virtual ~Date() {
        // 目前不需要做任何释放工作，因为构造函数没有申请资源
        cout << "析构函数" << endl;
    }
};

// 打印日期的实现
void Date::print() {
    cout << y << "-" << m << "-" << d << endl;
}

int main() {
    Date day;  // 创建一个Date对象，使用默认构造函数
    day.print();  // 打印日期
    return 0;  // 返回0表示程序正常结束
}
```
## 20.类模板
我们可以将一个类变成“类模板”或“模板类”，正如一个模板函数一样。
```C++
#include <iostream>
#include <cstdlib>
using namespace std;

// Array类模板，用于动态数组管理
template<class T>
class Array {
    int size;  // 数组大小
    T* data;   // 数组数据指针

public:
    // 构造函数，初始化数组大小并分配内存
    Array(int s) : size(s), data(new T[s]) {}

    // 虚析构函数，确保派生类对象能够正确释放资源
    virtual ~Array() {
        delete[] data;  // 释放动态分配的内存
    }

    // 重载[]运算符，提供数组访问功能
    T& operator[](int i) {
        if (i < 0 || i >= size) {
            cerr << "Error: Index out of bounds!" << endl;  // 输出错误信息
            throw out_of_range("Index out of range");       // 抛出异常
        }
        return data[i];  // 返回数组元素
    }
};

int main()
{
    try {
        // 测试int类型的数组
        Array<int> t(5);
        t[0] = 45;          // OK
        t[4] = t[0] + 6;    // OK
        cout << t[4] << endl;  // 输出51

        t[10] = 7;  // 错误：索引越界，抛出异常
    }
    catch (const out_of_range& e) {
        cerr << "Caught exception: " << e.what() << endl;
    }

    try {
        // 测试double类型的数组
        Array<double> a(5);
        a[0] = 45.5;        // OK
        a[4] = a[0] + 6.5;  // OK
        cout << a[4] << endl;  // 输出52.0

        a[10] = 7.5;  // 错误：索引越界，抛出异常
    }
    catch (const out_of_range& e) {
        cerr << "Caught exception: " << e.what() << endl;
    }

    return 0;  // 程序正常结束
}
```
## 21.typedef 类型别名
```C++
#include <iostream>
using namespace std;
typedef int INT;
int main() {
    INT i = 3; //等价于int i = 3;
    cout << i << endl;
    return 0;
}
```
## 22.String

```C++
#include <iostream>
#include <string> // typedef std::basic_string<char> string;
using namespace std;

typedef string String; // 定义String为string的别名

int main() {
    // 默认构造函数：没有参数或参数有默认值
    string s1; // s1为空字符串
    String s2("hello"); // 普通构造函数，String是string的别名
    s1 = "Anatoliy"; // 使用赋值运算符为s1赋值
    String s3(s1); // 拷贝构造函数，等价于 string s3 = s1;

    // 输出s1, s2, s3的值
    cout << "s1 is: " << s1 << endl;
    cout << "s2 is: " << s2 << endl;
    cout << "s3 is: " << s3 << endl; // 修正了原代码中的错误，s3应为s1的拷贝

    // 使用C字符串和字符数初始化
    string s4("this is a C_sting", 10); // 从C字符串中取前10个字符
    cout << "s4 is: " << s4 << endl;

    // 使用C++字符串、起始位置和字符数初始化
    string s5(s4, 6, 4); // 从s4的第6个字符开始，取4个字符
    cout << "s5 is: " << s5 << endl;

    // 使用字符数和字符本身初始化
    string s6(15, '*'); // 创建一个包含15个'*'的字符串
    cout << "s6 is: " << s6 << endl;

    // 使用迭代器范围初始化
    string s7(s4.begin(), s4.end() - 5); // 从s4的开头到倒数第5个字符
    cout << "s7 is: " << s7 << endl;

    // 使用赋值初始化
    string s8 = "Anatoliy"; // 直接赋值初始化
    cout << "s8 is: " << s8 << endl;

    // 字符串拼接
    string s9 = s1 + " hello " + s2; // s1 + " hello " + s2的结果是string类型的对象
    cout << "s9 is: " << s9 << endl;

    return 0; // 程序正常结束
}
```
遍历数组
```C++

#include <iostream>
#include <string>
using namespace std;

int main() {
    string s = "hell";  // 初始化字符串s
    string w = "worl!"; // 初始化字符串w

    s = s + w; // 将w拼接到s的末尾，等价于 s += w;

    // 使用下标访问并遍历字符串
    for (int ii = 0; ii != s.size(); ii++) {
        cout << ii << " " << s[ii] << endl; // 输出索引和对应的字符
    }
    cout << endl;

    // 使用迭代器遍历字符串
    string::const_iterator cii; // 定义常量迭代器
    int ii = 0; // 用于记录索引
    for (cii = s.begin(); cii != s.end(); cii++) {
        cout << ii++ << " " << *cii << endl; // 输出索引和对应的字符
    }
    //普通迭代器可以修改数值
    string::iterator cii; // 定义常量迭代器
    int ii = 0; // 用于记录索引
    for (cii = s.begin(); cii != s.end(); cii++) {
        *cii = 'A';
        cout << ii++ << " " << *cii << endl; // 输出索引和对应的字符
    }
    return 0; // 程序正常结束
}
```

## 23.Vector
```C++
#include <vector>
#include <iostream>
using namespace std;

int main() {
    vector<double> student_marks; // 存储学生成绩的向量
    int num_students; // 学生人数

    // 输入学生人数
    cout << "Number of students: " << endl;
    cin >> num_students;

    // 调整向量大小以容纳所有学生成绩
    student_marks.resize(num_students);

    // 输入每个学生的成绩
    for (vector<double>::size_type i = 0; i < num_students; i++) {
        cout << "Enter marks for student #" << i + 1 << ": " << endl;
        cin >> student_marks[i];
    }

    cout << endl;

    // 输出所有学生的成绩
    for (vector<double>::iterator it = student_marks.begin()
         it != student_marks.end(); it++) {
        cout << *it << endl;
    }

    return 0; // 程序正常结束
}
```

## 24.Inheritance继承(Derivation派生)
Inheritance继承(Derivation派生)： 一个派生类(derived class)从1个或多个父类(parent class) / 基类(base class)继承，即继承父类的属性和行为，但也有自己的特有属性和行为。如：
```C++
#include <iostream>
#include <string>
using namespace std;

// 定义Employee类
class Employee {
    string name; // 员工姓名
public:
    Employee(string n); // 构造函数
    void print(); // 打印员工信息
};

// 定义Manager类，继承自Employee
class Manager : public Employee {
    int level; // 管理级别
public:
    Manager(string n, int l = 1); // 构造函数，管理级别默认为1
    void print(); // 打印经理信息
};

// Employee类的构造函数，使用初始化列表初始化成员变量
Employee::Employee(string n) : name(n) {
    // 使用初始化列表比在构造函数体内赋值更高效
}

// 打印Employee的信息
void Employee::print() {
    cout << name << endl;
}

// Manager类的构造函数，使用初始化列表初始化基类和派生类的成员变量
Manager::Manager(string n, int l) : Employee(n), level(l) {
    // 派生类的构造函数只能初始化它自己的成员和直接基类的成员
}

// 错误：Manager类的构造函数不能直接初始化基类的成员变量name
// Manager::Manager(string n, int l) : name(n), level(l) {
//     // 这行代码是错误的，因为name是基类的私有成员，不能在派生类中直接初始化
// }

// 打印Manager的信息
void Manager::print() {
    cout << level << "\t"; // 打印管理级别
    Employee::print(); // 调用基类的print函数打印姓名
}

int main() {
    Manager m("Zhang", 2); // 创建一个Manager对象，管理级别为2
    Employee e("Li"); // 创建一个Employee对象

    m.print(); // 打印Manager的信息
    e.print(); // 打印Employee的信息

    return 0;
}
```

## 24.类和面向对象
### 1)类的定义方法
```C++
class className
{
    //成员函数
    //成员变量
};
```
简单来说 类就是属性和方法的集合，属性就是类中的数据，方法就是调用这些数据进行操作的函数
A:新建一个CPP文件，函数和成员变量都放到一个文件里
B:定义和声明分开放  ----封装

### 2)类的访问权限
限定访问符 private, protected public 可以在类外被访问


### 3)封装
封装：本质是一种管理手段，将属性（数据）和方法（接口）有机结合起来，再隐藏他们，只对外公开接口来和对象进行交互

**面向对象三大特性，封装，继承，多态**
```C++
class Stack
{
public:
    void Init(int capacity=4)

private: //数据私有
    int* _arr;
    int _length;
    int _capacity;
}

//stack.cpp
void Stack::Init(int capacity)
{
    _arr=(*int)malloc(sizeof(int)*capacity);
    _length=0;
    _capacity=capacity;
};
//耦合程度很高
```

### 4)类的实例化
相当于使用这个类，具体对类进行定义
### 5)类对象模型
#1 如何计算类的大小
类中只计算成员变量的大小

#2 类对象的储存方式
成员函数不在类内,一个类可以定义不同对象 不同的对象数据不同但是方法都一样

空类，占据1个字节，声明他的存在
### 6)this指针
```C++
class Date
{
public:
    void SetDate(int year,int month,int day)
    {
        _year=year;
        _month=month;
        _day=day;
    }
private:
    int _year;
    int _month;
    int _day;
}

int main()
{
    Date d1;
    Date d2;
    d1.SetDate(2021,8,19);
    d1.SetDate(2021,2,20);
}
```
类中的成员函数是共用的，不存在于类中，可以理解为此时d1 d2都是被公用代码段中的Date所定义的，因此编译器无法使用SetDate区分d1 d2

解决办法：使用this指针
定义：编译器给每个非静态的成员函数增加了一个隐藏的指针函数，让该指针指向当前函数（在函数运行是调用该函数的对象），在函数体中所有成员变量的操作都是通过该指针访问
```C++
class Date
{
public:
    void SetDate( Date* this,int year,int month,int day)
    {
        //生成一个类指针
        this->_year=year;
        this->_month=month;
        this->_day=day;
    }
private:
    int _year;
    int _month;
    int _day;
}

int main()
{
    Date d1;
    Date d2;
    d1.SetDate(&d1,2021,8,19);
    d1.SetDate(&d2,2021,2,20);
} 
```
