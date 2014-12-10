#include <iostream>
#include "cuda_runtime.h"
#include "..\include\opencv\highgui.h"

using namespace std;

__host__ void run_kernel(
	uchar4 *img,
	int size,
	uchar4 color1,
	uchar4 color2,
	int grid_step,
	uchar4 grid_color,
	int2 circ_pos,
	double radius,
	uchar4 circ_color);

int N = 800;
static uchar4 color1 = {255, 0, 0};
static uchar4 color2 = {0, 255, 0};
int grid_step = 20;
static uchar4 grid_color = {0, 255, 0};
int2 circ_pos = {100, 200};
double radius = 50;
static uchar4 circ_color = {0, 0, 0};

void otherVal()
{
	cin.clear();
	cin.ignore(cin.rdbuf()->in_avail(), '\n');
	cout << "Zadej velikost obrazku: ";
	cin >> N;
	while (!cin.good())
	{
		cin.clear();
		cin.ignore(cin.rdbuf()->in_avail(), '\n');
		cout << "Neplatna hodnota!!!" << endl << "Zadej velikost obrazku (cele cislo): ";
		cin >> N;
	}
	cin.clear();
	cin.ignore(cin.rdbuf()->in_avail(), '\n');
	cout << "Zadej rozpeti mrizky: ";
	cin >> grid_step;
	while (!cin.good())
	{
		cin.clear();
		cin.ignore(cin.rdbuf()->in_avail(), '\n');
		cout << "Neplatna hodnota!!!" << endl << "Zadej rozpeti mrizky (cele cislo): ";
		cin >> grid_step;
	}
	cin.clear();
	cin.ignore(cin.rdbuf()->in_avail(), '\n');
	cout << "Zadej polomer kruznice: ";
	cin >> radius;
	while (!cin.good())
	{
		cin.clear();
		cin.ignore(cin.rdbuf()->in_avail(), '\n');
		cout << "Neplatna hodnota!!!" << endl << "Zadej polomer kruznice (desetinne cislo): ";
		cin >> radius;
	}
	cin.clear();
	cin.ignore(cin.rdbuf()->in_avail(), '\n');
	cout << "Zadej pozici stredu kruznice:" << endl << "\tx = ";
	cin >> circ_pos.x;
	while (!cin.good())
	{
		cin.clear();
		cin.ignore(cin.rdbuf()->in_avail(), '\n');
		cout << "Neplatna hodnota!!!" << endl << "\tx = ";
		cin >> circ_pos.x;
	}
	cin.clear();
	cin.ignore(cin.rdbuf()->in_avail(), '\n');
	cout << "\ty = ";
	cin >> circ_pos.y;
	while (!cin.good())
	{
		cin.clear();
		cin.ignore(cin.rdbuf()->in_avail(), '\n');
		cout << "Neplatna hodnota!!!" << endl << "\ty = ";
		cin >> circ_pos.y;
	}
}

int main(void)
{
	char c;
	cout << "Vychozi hodnoty? (y/n): ";
	cin >> c;
	while (!cin.good() || (c != 'y' && c != 'n' && c != 'Y' && c != 'N'))
	{
		cin.clear();
		cin.ignore(cin.rdbuf()->in_avail(), '\n');
		cout << "Zadej 'y' nebo 'n': ";
		cin >> c;
	}
	if (c == 'y' || c == 'Y')
	{
		otherVal();
	}
	
	uchar4 *bgr_pole = new uchar4[N * N];
	IplImage *img = cvCreateImage(cvSize(N, N), IPL_DEPTH_8U, 3);

	for (int y = 0; y < N; y++)
	{
		for (int x = 0; x < N; x++)
		{
			uchar4 bgr = {255, 255, 255};
			bgr_pole[y * N + x] = bgr;

			CvScalar s = {
				bgr.x,
				bgr.y,
				bgr.z
			};

			cvSet2D(img, y, x, s);
		}
	}

	run_kernel(bgr_pole, N, color1, color2, grid_step, grid_color, circ_pos, radius, circ_color);

	IplImage *final_img = cvCreateImage(cvSize(N, N), IPL_DEPTH_8U, 3);

	for (int y = 0; y < N; y++)
	{
		for (int x = 0; x < N; x++)
		{
			uchar4 bgr = bgr_pole[y * N + x];

			CvScalar s = {
				bgr.x,
				bgr.y,
				bgr.z
			};

			cvSet2D(final_img, y, x, s);
		}
	}

	cvShowImage("Final image", final_img);

	cvWaitKey(0);
	return 0;
}