#include "../include/BMP.hpp"
#include <iostream>

BMP::BMP(const char *filename, int width, int height)
{
    this->filename = filename;

    this->file.open(filename, std::ios::binary | std::ios::out);
    if (!this->file)
    {
        throw std::runtime_error("Could not open file for writing");
    }

    this->width = width;
    this->height = height;
    this->rowSizeBytes = ((width * BYTES_PER_PIXEL + 3) & ~3); // Row size must be a multiple of 4 bytes

    // We are storing each color of the pixels as an 8-bit unsigned integer (0-255)
    this->pixelData = new uint8_t[rowSizeBytes * height];
    if (!this->pixelData)
    {
        throw std::runtime_error("Could not allocate memory for pixel data");
    }

    // Set up the BMP file header
    this->header.bfType = 0x4D42; // 'BM'
    this->header.bfSize = sizeof(HEADER) + sizeof(INFOHEADER) + (rowSizeBytes * height);
    this->header.bfReserved1 = 0;
    this->header.bfReserved2 = 0;
    this->header.bfOffBits = sizeof(HEADER) + sizeof(INFOHEADER);

    // Set up the BMP info header
    this->infoHeader.biSize = sizeof(INFOHEADER);
    this->infoHeader.biWidth = width;
    this->infoHeader.biHeight = height;
    this->infoHeader.biPlanes = 1;
    this->infoHeader.biBitCount = 24;        // 24 bits per pixel (RGB)
    this->infoHeader.biCompression = 0;      // No compression
    this->infoHeader.biSizeImage = 0;        // Can be zero for uncompressed images
    this->infoHeader.biXPelsPerMeter = 2835; // 72 DPI in PPM
    this->infoHeader.biYPelsPerMeter = 2835; // 72 DPI in PPM
    this->infoHeader.biClrUsed = 0;          // All colors are important
    this->infoHeader.biClrImportant = 0;     // All colors are important

    // Print info to console
    std::cout << "BMP file header size: " << sizeof(HEADER) << " bytes" << std::endl;
    std::cout << "BMP info header size: " << sizeof(INFOHEADER) << " bytes" << std::endl;
    std::cout << "BMP file size: " << this->header.bfSize << " bytes" << std::endl;
    std::cout << "BMP image row size: " << rowSizeBytes << " bytes" << std::endl;
}

BMP::~BMP()
{
    delete[] pixelData;
    if (file.is_open())
    {
        file.close();
    }
}

void BMP::writePixel(int x, int y, uint8_t r, uint8_t g, uint8_t b)
{
    if (x < 0 || x >= width || y < 0 || y >= height)
    {
        throw std::out_of_range("Pixel coordinates out of bounds");
    }

    int index = ((height - 1 - y) * width + x) * BYTES_PER_PIXEL; // BMP stores pixels bottom-up

    // Storing in BGR format
    pixelData[index] = b;     // Blue
    pixelData[index + 1] = g; // Green
    pixelData[index + 2] = r; // Red
}

void BMP::save()
{
    // Write the headers to the file
    this->file.write((char *)&this->header, sizeof(HEADER));
    this->file.write((char *)&this->infoHeader, sizeof(INFOHEADER));

    this->file.seekp(this->header.bfOffBits, std::ios::beg); // Move to the pixel data offset

    for (int y = 0; y < height; y++)
    {
        this->file.write((char *)&this->pixelData[y * this->rowSizeBytes], this->rowSizeBytes);
    }

    std::cout << "BMP file saved: " << this->filename << std::endl;
}