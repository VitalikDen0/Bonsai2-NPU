
float test_conv(unsigned short h) {
    __fp16 x;
    __builtin_memcpy(&x, &h, 2);
    return (float)x;
}
