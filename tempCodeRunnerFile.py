from decimal import Decimal, getcontext

getcontext().prec = 500

x = Decimal(input('输入未知数: '))

y = 0

for i in range(100000):
    z = i * x**(i-1)
    y += z

print(y)
