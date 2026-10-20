#include <cstdio>
#include <cstdint>
#include <cstddef>

// 1. alignas(1)：只影响结构体整体对齐，成员之间仍有 padding
struct alignas(16) Alignas1 {
    uint8_t  a;   // offset 0
    // 3 bytes padding
    uint32_t b;   // offset 4
};

// 2. #pragma pack(1)：成员最大对齐限制为 1，消除 padding
#pragma pack(push, 1)
struct PackedPragma {
    uint8_t  a;   // offset 0
    uint32_t b;   // offset 1
};
#pragma pack(pop)

// 3. __attribute__((packed))：GCC/Clang 下消除 padding
struct __attribute__((packed)) PackedAttr {
    uint8_t  a;   // offset 0
    uint32_t b;   // offset 1
};

// 4. 默认对齐：成员按自然对齐排列
struct Default {
    uint8_t  a;   // offset 0
    // 3 bytes padding
    uint32_t b;   // offset 4
};

// 5. 嵌套结构体：alignas(1) 不会消除内部成员的 padding
struct alignas(1) NestedAlignas1 {
    uint8_t  tag;          // offset 0
    // 3 bytes padding
    uint32_t value;        // offset 4
    // 4 bytes padding（为了让嵌套结构体整体是 8 的倍数）
};

// 6. 嵌套结构体：#pragma pack(1) 会递归消除所有 padding
#pragma pack(push, 1)
struct NestedPacked {
    uint8_t  tag;          // offset 0
    uint32_t value;        // offset 1
};
#pragma pack(pop)

// 7. 实际使用场景：网络协议头 / 二进制序列化
#pragma pack(push, 1)
struct PacketHeader {
    uint16_t type;         // offset 0
    uint16_t seq;          // offset 2
    uint32_t length;       // offset 4
    uint8_t  flags;        // offset 8
};
#pragma pack(pop)

int main() {
    printf("=== 基本对比 ===\n");
    printf("Default:        sizeof=%zu, alignof=%zu\n",
           sizeof(Default), alignof(Default));
    printf("alignas(1):     sizeof=%zu, alignof=%zu\n",
           sizeof(Alignas1), alignof(Alignas1));
    printf("#pragma pack(1):sizeof=%zu, alignof=%zu\n",
           sizeof(PackedPragma), alignof(PackedPragma));
    printf("__packed__:     sizeof=%zu, alignof=%zu\n",
           sizeof(PackedAttr), alignof(PackedAttr));

    printf("\n=== 嵌套结构体 ===\n");
    printf("Nested alignas(1):  sizeof=%zu, alignof=%zu\n",
           sizeof(NestedAlignas1), alignof(NestedAlignas1));
    printf("Nested #pragma:     sizeof=%zu, alignof=%zu\n",
           sizeof(NestedPacked), alignof(NestedPacked));

    printf("\n=== 协议头示例 ===\n");
    printf("PacketHeader:   sizeof=%zu\n", sizeof(PacketHeader));

    // 验证成员偏移
    printf("\n=== 成员偏移验证 ===\n");
    printf("PacketHeader.type:   offset %zu\n", offsetof(PacketHeader, type));
    printf("PacketHeader.seq:    offset %zu\n", offsetof(PacketHeader, seq));
    printf("PacketHeader.length: offset %zu\n", offsetof(PacketHeader, length));
    printf("PacketHeader.flags:  offset %zu\n", offsetof(PacketHeader, flags));

    // 验证 alignas(1) 的作用：整体对齐为 1
    printf("\n=== alignas(1) 的实际效果 ===\n");
    Alignas1 obj1;
    printf("obj1 地址: %p, 是否 4 字节对齐: %s\n",
           (void*)&obj1,
           ((uintptr_t)&obj1.a % 4 == 0) ? "是" : "否");
    printf("obj1.a 地址: %p, 是否 4 字节对齐: %s\n",
           (void*)&obj1.a,
           ((uintptr_t)&obj1.a % 4 == 0) ? "是" : "否");
    printf("obj1.b 地址: %p, 是否 4 字节对齐: %s\n",
           (void*)&obj1.b,
           ((uintptr_t)&obj1.b % 4 == 0) ? "是" : "否");

    // 在数组中验证：alignas(1) 允许非对齐起始
    Alignas1 arr[2];
    printf("arr[0] 地址: %p\n", (void*)&arr[0]);
    printf("arr[1] 地址: %p, 间距: %zu\n",
           (void*)&arr[1],
           (uintptr_t)&arr[1] - (uintptr_t)&arr[0]);

    // 对比 #pragma pack(1) 的数组间距
    PackedPragma arr2[2];
    printf("\n#pragma pack(1) 数组间距: %zu\n",
           (uintptr_t)&arr2[1] - (uintptr_t)&arr2[0]);

    return 0;
}