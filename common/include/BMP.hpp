#include <stdint.h>
#include <fstream>

class BMP
{
private:
    const int BITS_PER_PIXEL = 24;                  // 24 bits per pixel (RGB)
    const int BYTES_PER_PIXEL = BITS_PER_PIXEL / 8; // 3 bytes per pixel (RGB)

    std::ofstream file;
    const char *filename;
    uint8_t *pixelData;
    int height;
    int width;
    int rowSizeBytes; // Size of each row in bytes (including padding)

    #pragma pack(push, 1)
    struct HEADER
    {
        uint16_t bfType;      // Magic number for BMP files
        uint32_t bfSize;      // Size of the BMP file in bytes
        uint16_t bfReserved1; // Reserved; must be 0
        uint16_t bfReserved2; // Reserved; must be 0
        uint32_t bfOffBits;   // Offset to start of pixel data
    } header;

    struct INFOHEADER
    {
        uint32_t biSize;         // Size of this header (40 bytes)
        int32_t biWidth;         // Width of the bitmap in pixels
        int32_t biHeight;        // Height of the bitmap in pixels
        uint16_t biPlanes;       // Number of color planes (must be 1)
        uint16_t biBitCount;     // Bits per pixel (1, 4, 8, or 24)
        uint32_t biCompression;  // Compression type (0 = none)
        uint32_t biSizeImage;    // Size of the image data in bytes
        int32_t biXPelsPerMeter; // Horizontal resolution (pixels per meter)
        int32_t biYPelsPerMeter; // Vertical resolution (pixels per meter)
        uint32_t biClrUsed;      // Number of colors in the color palette
        uint32_t biClrImportant; // Important colors (0 = all)
    } infoHeader;
    #pragma pack(pop)

public:
    BMP(const char *filename, int width, int height);
    ~BMP();

    void writePixel(int x, int y, uint8_t r, uint8_t g, uint8_t b);
    void save();
};