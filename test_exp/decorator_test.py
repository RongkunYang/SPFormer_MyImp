def wrapperfun(func):
    def warper(*args, **kwargs):
        print(func(*args, **kwargs))
    return warper


@wrapperfun
def say_hello(name):
    return 'HI ' + name

say_hello('Yang')


#funA 作为装饰器函数
def funA(fn):
    print("C语言中文网")
    fn() # 执行传入的fn参数
    print("http://c.biancheng.net")
    return "装饰器函数的返回值"

@funA
def funB():
    print("学习 Python")

print(funB)
# 因为这里是直接将funA的返回值赋值给funB的，所以在调用funB的时候不用加括号，
# 为了满足一般的函数调用形式，所以装饰器函数一般返回的是一个函数对象。
