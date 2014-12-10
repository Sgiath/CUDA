#include <iostream>
#include "cuda_runtime.h"
#include "..\include\opencv\highgui.h"

using namespace std;

__host__ void run_kernel_cut(uchar4 *orig, uchar4 *fin, int2 img_dim, int2 new_image_dim, int4 cut);
__host__ int run_kernel_turn(uchar4 *orig, uchar4 *fin, int2 img_dim, int2 new_img_dim, int turn);
__host__ void run_kernel_median(uchar4 *orig, uchar4 *fin, int2 img_dim, int neigh);
__host__ void run_kernel_invert(uchar4 *orig, uchar4 *fin, int2 img_dim);
__host__ void run_kernel_resize(uchar4 *orig, uchar4 *fin, int2 img_dim, int2 new_img_dim, int index);

void cut(uchar4 *orig, uchar4 *fin, int2 size, int2 new_size, int4 cut)
{
	
	run_kernel_cut(orig, fin, size, new_size, cut);

	return;
}

void turn(uchar4 *orig, uchar4 *fin, int2 size, int2 new_size, int turn)
{
	run_kernel_turn(orig, fin, size, new_size, turn);

	return;
}

void median(uchar4 *orig, uchar4 *fin, int2 img_dim, int neigh)
{
	run_kernel_median(orig, fin, img_dim, neigh);

	return;
}

void invert(uchar4 *orig, uchar4 *fin, int2 img_dim)
{
	run_kernel_invert(orig, fin, img_dim);

	return;
}

void resize(uchar4 *orig, uchar4 *fin, int2 img_dim, int2 new_img_dim, int index)
{
	
	run_kernel_resize(orig, fin, img_dim, new_img_dim, index);

	return;
}

int main(int numarg, char **arg)
{
	string obr;
	if (numarg < 2)
	{
		cout << "Zadej cestu k souboru: ";
		cin >> obr;
		while (!cin.good())
		{
			cin.clear();
			cin.ignore(cin.rdbuf()->in_avail(), '\n');
			cout << "Neplatny vstup!!!" << endl << "Zadej cestu k souboru: ";
			cin >> obr;
		}
	}
	else
	{
		obr = arg[1];
	}
	IplImage *orig_img = cvLoadImage(obr.c_str());

	while (orig_img == nullptr);
	{
		cin.clear();
		cin.ignore(cin.rdbuf()->in_avail(), '\n');
		cout << "Nelze otevrit soubor!!!" << endl << "Zadej cestu k souboru: ";
		cin >> obr;
		while (!cin.good())
		{
			cin.clear();
			cin.ignore(cin.rdbuf()->in_avail(), '\n');
			cout << "Neplatny vstup!!!" << endl << "Zadej cestu k souboru: ";
			cin >> obr;
		}
		orig_img = cvLoadImage(obr.c_str());
	}

	int2 size = {
		orig_img->width,
		orig_img->height
	};

	uchar4 *bgr_orig_image = new uchar4[size.x * size.y];

	for (int y = 0; y < size.y; y++)
	{
		for (int x = 0; x < size.x; x++)
		{
			CvScalar s = cvGet2D(orig_img, y, x);
			uchar4 bgr = {
				(char)s.val[0],
				(char)s.val[1],
				(char)s.val[2],
				0
			};
			bgr_orig_image[y * size.x + x] = bgr;
		}
	}

	int2 new_size = size;
	uchar4 *bgr_final_image;
	int choice;

	cout << endl << "Pro oriznuti obrazku zadejete 1," << endl
		<< "pro otoceni zadejte 2," << endl
		<< "pro medianovou filtraci zadejte 3," << endl
		<< "pro invertovani barev zadejte 4," << endl
		<< "pro zmenu velikosti zadejte 5: ";
	cin >> choice;
	while (choice != 1 && choice != 2 && choice != 3 && choice != 4 && choice != 5)
	{
		cout << endl << "Neplatna volba, zadej znovu: ";
		cin >> choice;
	}

	if (choice == 1)
	{
		int4 cut_dim;
		cout << "Oriznuti vlevo: ";
		cin >> cut_dim.x;
		cout << "Oriznuti vpravo: ";
		cin >> cut_dim.y;
		if (cut_dim.x + cut_dim.y >= size.x)
		{
			cout << "Prilis velke oriznuti, je pouzito vychozi (0, 0)";
			cut_dim.x = cut_dim.y = 0;
		}
		cout << "Oriznuti nahore: ";
		cin >> cut_dim.z;
		cout << "Oriznuti dole: ";
		cin >> cut_dim.w;
		if (cut_dim.z + cut_dim.w >= size.y)
		{
			cout << "Prilis velke oriznuti, je pouzito vychozi (0, 0)";
			cut_dim.z = cut_dim.w = 0;
		}

		new_size.x = size.x - (cut_dim.x + cut_dim.y);
		new_size.y = size.y - (cut_dim.z + cut_dim.w);

		bgr_final_image = new uchar4[new_size.x * new_size.y];

		cut(bgr_orig_image, bgr_final_image, size, new_size, cut_dim);
	}
	else if (choice == 2)
	{
		int ch_turn;
		cout << "Pro otoceni vpravo zadejte 1," << endl
			<< "pro otoceni vzhuru nohama 2" << endl 
			<< "a pro otoceni vlevo zadejte 3: ";
		cin >> ch_turn;
		if (ch_turn != 1 && ch_turn != 2 && ch_turn != 3)
		{
			cout << "Neplatne otoceni, je pouzito vychozi (right)";
			ch_turn = 1;
		}

		if (ch_turn == 1 || ch_turn == 3)
		{
			new_size.x = size.y;
			new_size.y = size.x;
		}
		else
		{
			new_size = size;
		}
		bgr_final_image = new uchar4[new_size.x * new_size.y];
		turn(bgr_orig_image, bgr_final_image, size, new_size, ch_turn);
	}
	else if (choice == 3)
	{
		int neigh;
		cout << endl << "Zadejte velikost okoli pixelu: ";
		cin >> neigh;
		bgr_final_image = new uchar4[new_size.x * new_size.y];
		median(bgr_orig_image, bgr_final_image, new_size, neigh);
	}
	else if (choice == 4)
	{
		bgr_final_image = new uchar4[new_size.x * new_size.y];
		invert(bgr_orig_image, bgr_final_image, new_size);
	}
	else if (choice == 5)
	{
		int index;
		cout << endl << "Zadejte parametr zmeny velikosti obrazku: ";
		cin >> index;

		new_size.x = size.x * index;
		new_size.y = size.y * index;

		bgr_final_image = new uchar4[new_size.x * new_size.y];

		for (int i = 0; i < new_size.x * new_size.y; i++)
		{
			bgr_final_image[i] = {0, 0, 0, 1};
		}

		resize(bgr_orig_image, bgr_final_image, size, new_size, index);
	}

	IplImage *final_img = cvCreateImage(cvSize(new_size.x, new_size.y), IPL_DEPTH_8U, 3);

	for (int y = 0; y < new_size.y; y++)
		for (int x = 0; x < new_size.x; x++)
		{
			uchar4 bgr = bgr_final_image[y * new_size.x + x];

			CvScalar s = {
				bgr.x,
				bgr.y,
				bgr.z
			};

			cvSet2D(final_img, y, x, s);
		}

	cvShowImage("Original Image", orig_img);
	cvShowImage("Edited Image", final_img);
	cvWaitKey(0);
	return 0;
}

