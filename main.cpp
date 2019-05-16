/* 
 * File:   main.cpp
 * Author: gregw
 *
 * 
 * Exercises for chapter 16 of Learning OpenCV3, Keypoints and Descriptors.
 * Most of this code started as samples from the OpenCV library, that were then
 * modified for specific exercises.
 * 
 * Created on March 9, 2019, 9:45 AM
 */

#include <cstdlib>
#include <iostream>
#include <dirent.h>
#include <math.h>

#include <opencv2/opencv.hpp>

#include "funcs.h"

using namespace std;
using namespace cv;


void exer16_3(int, char**);
void exer16_6(int, char**);
void exer16_8(string);
void exer16_9(int, char**);


/*
 * 
 */
int main(int argc, char** argv)
{

    exer16_3(argc, argv);
    //exer16_6(argc, argv);
    //exer16_8(argv[1]);
    //exer16_9(argc, argv); 
    
    return 0;
}


static void help()
{
    cout << "\n This program demonstrates how to detect compute and match ORB BRISK and AKAZE descriptors \n"
        "Usage: \n"
        "  ./matchmethod_orb_akaze_brisk --image1=<image1(basketball1.png as default)> --image2=<image2(basketball2.png as default)>\n"
        "Press a key when image window is active to change algorithm or descriptor";
}


// Starting with sample matchmethod_orb_akaze_brisk.cpp from OpenCV, train on a planar object, then see how well it
// tracks it. Thus, want to take image of a book to train on, then video of that book. Function has to be modified
// to go through all the images in a video before moving to the next method. Used book_a.JPG and book.MOV.
// Receives: Name of file containing object to track, name of video file to track object in.
// Note that is hardcoded to show max 40 frames of the video, as it can get tedious - adjust loop if want to see more.
void exer16_3(int argc, char **argv)
{
    vector<String> typeDesc;
    vector<String> typeAlgoMatch;
    vector<String> fileName;
    // This descriptor are going to be detect and compute
    typeDesc.push_back("AKAZE-DESCRIPTOR_KAZE_UPRIGHT");    // see http://docs.opencv.org/trunk/d8/d30/classcv_1_1AKAZE.html
    typeDesc.push_back("AKAZE");    // see http://docs.opencv.org/trunk/d8/d30/classcv_1_1AKAZE.html
    typeDesc.push_back("ORB");      // see http://docs.opencv.org/trunk/de/dbf/classcv_1_1BRISK.html
    typeDesc.push_back("BRISK");    // see http://docs.opencv.org/trunk/db/d95/classcv_1_1ORB.html
    // This algorithm would be used to match descriptors see http://docs.opencv.org/trunk/db/d39/classcv_1_1DescriptorMatcher.html#ab5dc5036569ecc8d47565007fa518257
    typeAlgoMatch.push_back("BruteForce");
    typeAlgoMatch.push_back("BruteForce-L1");
    typeAlgoMatch.push_back("BruteForce-Hamming");
    typeAlgoMatch.push_back("BruteForce-Hamming(2)");
    cv::CommandLineParser parser(argc, argv,
        "{ @image1 | basketball1.png | }"
        "{ @image2 | basketball2.png | }"
        "{help h ||}");
    if (parser.has("help"))
    {
        help();
        return;
    }
    fileName.push_back(samples::findFile(parser.get<string>(0)));
    fileName.push_back(samples::findFile(parser.get<string>(1)));
    Mat img1 = get_small_image(fileName[0], IMREAD_GRAYSCALE);
    //Mat img1 = imread(fileName[0], IMREAD_GRAYSCALE);
    Mat img2, gray2;   // = imread(fileName[1], IMREAD_GRAYSCALE);
    VideoCapture cap;
    cap.open(fileName[1]);
    if (img1.empty())
    {
        cerr << "Image " << fileName[0] << " is empty or cannot be found" << endl;
        return;
    }
    if (!cap.isOpened())
    {
        cerr << "Video " << fileName[1] << " is empty or cannot be found" << endl;
        return;
    }

    vector<double> desMethCmp;
    Ptr<Feature2D> b;

    // Descriptor loop
    vector<String>::iterator itDesc;
    for (itDesc = typeDesc.begin(); itDesc != typeDesc.end(); ++itDesc)
    {
        Ptr<DescriptorMatcher> descriptorMatcher;
        // Match between img1 and img2
        vector<DMatch> matches;
        // keypoint  for img1 and img2
        vector<KeyPoint> keyImg1, keyImg2;
        // Descriptor for img1 and img2
        Mat descImg1, descImg2;
        vector<String>::iterator itMatcher = typeAlgoMatch.end();
        if (*itDesc == "AKAZE-DESCRIPTOR_KAZE_UPRIGHT"){
            b = AKAZE::create(AKAZE::DESCRIPTOR_KAZE_UPRIGHT);
        }
        if (*itDesc == "AKAZE"){
            b = AKAZE::create();
        }
        if (*itDesc == "ORB"){
            b = ORB::create();
        }
        else if (*itDesc == "BRISK"){
            b = BRISK::create();
        }
        try
        {
            // We can detect keypoint with detect method
            b->detect(img1, keyImg1, Mat());
            // and compute their descriptors with method  compute
            b->compute(img1, keyImg1, descImg1);
            // or detect and compute descriptors in one step

            for (itMatcher = typeAlgoMatch.begin(); itMatcher != typeAlgoMatch.end(); ++itMatcher){
                descriptorMatcher = DescriptorMatcher::create(*itMatcher);
                if ((*itMatcher == "BruteForce-Hamming" || *itMatcher == "BruteForce-Hamming(2)") && (b->descriptorType() == CV_32F || b->defaultNorm() <= NORM_L2SQR))
                {
                    cout << "**************************************************************************\n";
                    cout << "It's strange. You should use Hamming distance only for a binary descriptor\n";
                    cout << "**************************************************************************\n";
                }
                if ((*itMatcher == "BruteForce" || *itMatcher == "BruteForce-L1") && (b->defaultNorm() >= NORM_HAMMING))
                {
                    cout << "**************************************************************************\n";
                    cout << "It's strange. You shouldn't use L1 or L2 distance for a binary descriptor\n";
                    cout << "**************************************************************************\n";
                }


                vector<DMatch> bestMatches;
                cap.open(fileName[1]);
                for (int frame = 0; frame < 40; ++frame) {  // Only going through first 40 frames of video, while troubleshooting.
                    cap >> img2;
                    if (img2.empty()) break;
                    cvtColor(img2, gray2, COLOR_BGR2GRAY);
                    if (frame == 0) {
                        describe_mat(img2, "img2");
                        describe_mat(gray2, "gray2"); }
                    b->detectAndCompute(gray2, Mat(),keyImg2, descImg2,false);
                    // Match method loop
                    try
                    {
                        descriptorMatcher->match(descImg1, descImg2, matches, Mat());
                        // Keep best matches only to have a nice drawing.
                        // We sort distance between descriptor matches
                        Mat index;
                        int nbMatch=int(matches.size());
                        Mat tab(nbMatch, 1, CV_32F);
                        for (int i = 0; i<nbMatch; i++)
                        {
                            tab.at<float>(i, 0) = matches[i].distance;
                        }
                        sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
                        bestMatches.clear();
                        for (int i = 0; i<30; i++)
                        {
                            bestMatches.push_back(matches[index.at<int>(i, 0)]);
                        }
                        Mat result;
                        drawMatches(img1, keyImg1, gray2, keyImg2, bestMatches, result);
                        namedWindow(*itDesc+": "+*itMatcher, WINDOW_AUTOSIZE);
                        imshow(*itDesc + ": " + *itMatcher, result);
                        waitKey(40);
                    }
                    catch (const Exception& e)
                    {
                        cout << e.msg << endl;
                        cout << "Cumulative distance cannot be computed." << endl;
                        desMethCmp.push_back(-1);
                        break;  // Don't try rest of frames in video if this one failed.
                    }
                }       // End video loop.
                cap.release();
                        
                try
                {
                    // Saved result could be wrong due to bug 4308
                    FileStorage fs(*itDesc + "_" + *itMatcher + ".yml", FileStorage::WRITE);
                    fs<<"Matches"<<matches;
                    vector<DMatch>::iterator it;
                    cout<<"**********Match results**********\n";
                    cout << "Index \tIndex \tdistance\n";
                    cout << "in img1\tin img2\n";
                    // Use to compute distance between keyPoint matches and to evaluate match algorithm
                    double cumSumDist2=0;
                    for (it = bestMatches.begin(); it != bestMatches.end(); ++it)
                    {
                        cout << it->queryIdx << "\t" <<  it->trainIdx << "\t"  <<  it->distance << "\n";
                        Point2d p=keyImg1[it->queryIdx].pt-keyImg2[it->trainIdx].pt;
                        cumSumDist2=p.x*p.x+p.y*p.y;
                    }
                    desMethCmp.push_back(cumSumDist2);
                    waitKey();
                }
                catch (const Exception& e)
                {
                    cout << e.msg << endl;
                    cout << "Cumulative distance cannot be computed." << endl;
                    desMethCmp.push_back(-1);
                    break;      // Don't try rest of frames in video if this one failed.
                }
            }   
        }
        catch (const Exception& e)
        {
            cerr << "Exception: " << e.what() << endl;
            cout << "Feature : " << *itDesc << "\n";
            if (itMatcher != typeAlgoMatch.end())
            {
                cout << "Matcher : " << *itMatcher << "\n";
            }
        }
    }
    int i=0;
    cout << "Cumulative distance between keypoint match for different algorithm and feature detector \n\t";
    cout << "We cannot say which is the best but we can say results are different! \n\t";
    for (vector<String>::iterator itMatcher = typeAlgoMatch.begin(); itMatcher != typeAlgoMatch.end(); ++itMatcher)
    {
        cout<<*itMatcher<<"\t";
    }
    cout << "\n";
    for (itDesc = typeDesc.begin(); itDesc != typeDesc.end(); ++itDesc)
    {
        cout << *itDesc << "\t";
        for (vector<String>::iterator itMatcher = typeAlgoMatch.begin(); itMatcher != typeAlgoMatch.end(); ++itMatcher, ++i)
        {
            cout << desMethCmp[i]<<"\t";
        }
        cout<<"\n";
    }
    return;
}


static void help16_6()
{
    cout << "\n This program uses descriptors to try to match images in given directory to 3 specified training samples \n"
        "Usage: \n"
        "  ./exercises16 <image1> <image2> <image3> <dirName>\n";
}


Ptr<Feature2D> getMatcher(string matchName) {

    Ptr<Feature2D> b;
    if (matchName == "AKAZE-DESCRIPTOR_KAZE_UPRIGHT"){
        b = AKAZE::create(AKAZE::DESCRIPTOR_KAZE_UPRIGHT);
    } else if (matchName == "AKAZE"){
        b = AKAZE::create();
    } else if (matchName == "ORB"){
        b = ORB::create();
    } else if (matchName == "BRISK"){
        b = BRISK::create();
    }
    
    return b;
}


// Cop-out because getting list of files in a directory without using an external library seems nasty, non-portable.
vector<string> getTrainFiles(string dirName) {
    vector<string> myFiles;
    myFiles.push_back(dirName + "book_a_1.JPG");
    myFiles.push_back(dirName + "book_a_2.JPG");
    myFiles.push_back(dirName + "book_a_3.JPG");
    myFiles.push_back(dirName + "book_a_4.JPG");
    myFiles.push_back(dirName + "book_a_5.JPG");
    myFiles.push_back(dirName + "book_a_6.JPG");
    myFiles.push_back(dirName + "book_a_7.JPG");
    myFiles.push_back(dirName + "book_a_8.JPG");
    myFiles.push_back(dirName + "book_a_9.JPG");
    myFiles.push_back(dirName + "book_b_1.JPG");
    myFiles.push_back(dirName + "book_b_2.JPG");
    myFiles.push_back(dirName + "book_b_3.JPG");
    myFiles.push_back(dirName + "book_b_4.JPG");
    myFiles.push_back(dirName + "book_b_5.JPG");
    myFiles.push_back(dirName + "book_b_6.JPG");
    myFiles.push_back(dirName + "book_b_7.JPG");
    myFiles.push_back(dirName + "book_b_8.JPG");
    myFiles.push_back(dirName + "book_b_9.JPG");
    myFiles.push_back(dirName + "book_b_10.JPG");
    myFiles.push_back(dirName + "book_c_1.JPG");
    myFiles.push_back(dirName + "book_c_2.JPG");
    myFiles.push_back(dirName + "book_c_3.JPG");
    myFiles.push_back(dirName + "book_c_4.JPG");
    myFiles.push_back(dirName + "book_c_5.JPG");
    myFiles.push_back(dirName + "book_c_6.JPG");
    myFiles.push_back(dirName + "book_c_7.JPG");
    myFiles.push_back(dirName + "book_c_8.JPG");
    myFiles.push_back(dirName + "book_c_9.JPG");
    myFiles.push_back(dirName + "book_c_10.JPG");
    myFiles.push_back(dirName + "book_c_11.JPG");
    myFiles.push_back(dirName + "nobook_1.JPG");
    myFiles.push_back(dirName + "nobook_2.JPG");
    myFiles.push_back(dirName + "nobook_3.JPG");
    myFiles.push_back(dirName + "nobook_4.JPG");
    myFiles.push_back(dirName + "nobook_5.JPG");
    myFiles.push_back(dirName + "nobook_6.JPG");
    myFiles.push_back(dirName + "nobook_7.JPG");
    myFiles.push_back(dirName + "nobook_8.JPG");
    myFiles.push_back(dirName + "nobook_9.JPG");
    myFiles.push_back(dirName + "nobook_10.JPG");
    
    return myFiles;
}


double getImgDiff(vector<DMatch> bestMatches, vector<KeyPoint> sampleKeys, vector<KeyPoint> trainKeys) {

    double cumSumDist2=0;
    
    for (auto it = bestMatches.begin(); it != bestMatches.end(); ++it) {
        //cout << it->queryIdx << "\t" <<  it->trainIdx << "\t"  <<  it->distance << "\n";
        Point2d p = sampleKeys[it->queryIdx].pt - trainKeys[it->trainIdx].pt;
        cumSumDist2 += p.x*p.x+p.y*p.y;
    }
    
    return cumSumDist2;
}


// Starting with sample matchmethod_orb_akaze_brisk.cpp from OpenCV, create photos of 3 books against a blank background,
// store corresponding descriptors. Then take more photos of each book and some pictures with no books, try to detect
// the correct book or the fact that there is no book, for each of the additional pictures.
// Receives: Names of three image files that contain books against blank backgrounds and name of dir
// to find files to check. Expects sample book images to be in given directory.
// Output: For each file that it checks, either the name of the image that it feels is most likely in that image,
// or "no match found", and the difference measures between this image and the 3 exemplars.
// Create descriptors for each of the 3 books. Then, for each additional jpg file
// in given directory, calculate descriptors, differences from each of the sample books, decide whether the file
// matches a book or not.
// Will start using just one descriptor and one match algorithm.
// Results - correct matches, out of 41 images to test.
// Method      A   B   C  No Book  Total
// AK-Brute    7   10  8  5        30
// AK-Up-Brute 6   9   4  3        22
// Orb-Brute   6   9   7  6        28
// Brisk-Brute 7   10  5  8        30
void exer16_6(int argc, char **argv)
{
    vector<String> fileNames;
    //string typeDesc = "AKAZE";
    //string typeDesc = "AKAZE-DESCRIPTOR_KAZE_UPRIGHT";
    //string typeDesc = "ORB";
    string typeDesc = "BRISK";
    string typeAlgoMatch = "BruteForce";
    cv::CommandLineParser parser(argc, argv,
        "{ @image1 | basketball1.png | }"
        "{ @image2 | basketball2.png | }"
        "{help h ||}");
    if (parser.has("help")) {
        help16_6();
        return;
    }

    // Create descriptors for the 3 sample images.
    string dirname = argv[4];
    fileNames.push_back(dirname + argv[1]);
    fileNames.push_back(dirname + argv[2]);
    fileNames.push_back(dirname + argv[3]);
    vector<Mat> sampleImgs;
    vector<vector<KeyPoint>> sampleKeyImgs;
    vector<Mat> descriptorImgs;
    Ptr<Feature2D> b = getMatcher(typeDesc);
    int fnIdx = 0;
    for (auto it = fileNames.cbegin(); it != fileNames.cend(); ++it) {
        cout << "sample: " << fileNames[fnIdx] << endl;
        Mat img = get_small_image(*it, IMREAD_GRAYSCALE, false);
        vector<KeyPoint> tmpKI;
        cv::Mat tmpMat;
        b->detectAndCompute(img, Mat(), tmpKI, tmpMat, false);
        sampleImgs.push_back(img);
        sampleKeyImgs.push_back(tmpKI);
        descriptorImgs.push_back(tmpMat);
        ++fnIdx;
    }
    cout << endl << endl << "Starting images to check for matches." << endl;
    
    Mat trainImg;
    //vector<double> desMethCmp;
    Ptr<DescriptorMatcher> descriptorMatcher;
    vector<DMatch> matches;
    vector<KeyPoint> trainKeyImg;
    Mat trainDescImg;
    try
    {
        descriptorMatcher = DescriptorMatcher::create(typeAlgoMatch);


        // Decide whether each of the training files contains a book, and, if so, which one.
        vector<DMatch> bestMatches;
        vector<string> trainFiles = getTrainFiles(dirname);
        for (auto trainIt = trainFiles.cbegin(); trainIt != trainFiles.cend(); ++trainIt) {
            // Note that get_small_image finds a factor to scale image to under 800*600. Am assuming that all sample
            // and training images are the same size, so will end up same scale once read in.
            trainImg = get_small_image(*trainIt, IMREAD_GRAYSCALE, false);
            cout << *trainIt << ": ";
            b->detectAndCompute(trainImg, Mat(), trainKeyImg, trainDescImg, false);

            // Compare the training descriptors with with each of the three sample images.
            int bestGuess = 3;  //guess 0 = first pic, 1 = second, 2 = third, 3 = no match.
            double lowestDiff = 1700000;   // If no difference less than this, assume no match. (Value is an estimate of valid difference.)
            int sampleIdx = 0;
            for (auto sampleIt = descriptorImgs.cbegin(); sampleIt != descriptorImgs.cend(); ++sampleIt) {
                try {
                    descriptorMatcher->match(*sampleIt, trainDescImg, matches, Mat());
                    // Keep best matches only to have a nice drawing.
                    // We sort distance between descriptor matches
                    Mat index;
                    int nbMatch=int(matches.size());
                    Mat tab(nbMatch, 1, CV_32F);
                    //cout << ", nbMatch=" << nbMatch;
                    for (int i = 0; i<nbMatch; i++) {
                        tab.at<float>(i, 0) = matches[i].distance;
                    }
                    sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
                    bestMatches.clear();
                    for (int i = 0; i<30; i++) {
                        bestMatches.push_back(matches[index.at<int>(i, 0)]);
                    }
                    double thisDiff = getImgDiff(bestMatches, sampleKeyImgs[sampleIdx], trainKeyImg);
                    if( thisDiff < lowestDiff) {
                        bestGuess = sampleIdx;
                        lowestDiff = thisDiff;
                    }
                    cout << thisDiff << ", ";
                    //Mat result;
                    //drawMatches(sampleImgs[sampleIdx], sampleKeyImgs[sampleIdx], trainImg, trainKeyImg, bestMatches, result);
                    //string winName = fileNames[sampleIdx] + ": " + *trainIt;
                    //namedWindow(winName, WINDOW_AUTOSIZE);
                    //imshow(winName, result);
                    //waitKey();
                    //destroyAllWindows();
                }
                catch (const Exception& e) {
                    cout << e.msg << endl;
                    //cout << "Cumulative distance cannot be computed." << endl;
                    //desMethCmp.push_back(-1);
                    break;  // Don't try rest of frames in video if this one failed.
                }
                ++sampleIdx;
            }       // End sample image loop.
            if (bestGuess < 3) {
                cout << fileNames[bestGuess] << endl;
            }
            else {
                cout << " no match found." << endl;
            }
        }       // End of training files loop.
    }
        catch (const Exception& e) {
        cerr << "Exception: " << e.what() << endl;
    }

    return;
}


// Exercise 16.8 This exercise was implemented by modifying a copy of the OpenCV lkdemo project, 
// in a separate project, as it just involved a few changes to OpenCV sample code. 
// Tested with a video of my holding a squash in front of a blank wall and moving the squash around.
// One helper function here - load in first frame from video, display specified rectangle around squash,
// so can see if coordinates to select the object to track are correct.
// Results: For part A, removing cornerSubPix calls, seemed that points weren't as stable
// as when it was enabled - i.e. they disappeared more quickly.
// A:   At first couldn't get any points to stick, but turned out I was putting my grid of points
//      on the background, not the moving object. Once had them on the object, they seemed to work almost
//      as well as the found descriptors - did seem to drop off a bit faster.
void exer16_8(string videoName) {
 
    VideoCapture cap;
    cap.open(videoName);
    cv::Mat im1;
    cap >> im1;
    cv::rectangle(im1, cv::Point(600,250), cv::Point(1300, 700), cv::Scalar(0, 0, 255), 3);
    imshow("squash", im1);
    cv::waitKey(0);
    
    return;
}


// Helper class, to hold data passed to various functions for calculating optical flow.
struct FlowData {
public:
    Mat origGray;
    Mat newGray;
    vector<Point2f> origFeatures;
    vector<Point2f> newFeatures;
    vector<uchar> status;
    vector<float> err;

    // Stashing some constants for ease of access/visibility.
    TermCriteria termcrit = TermCriteria(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    Size subPixWinSize = Size(10,10);
    Size winSize = Size(31,31);
    int maxPoints = 100;

};


void getFeatures(const Mat & img, FlowData &myFData) {
    
    cvtColor(img, myFData.origGray, COLOR_BGR2GRAY);

    //cout << "size 0: " << points[0].size() << ", size 1: " << points[1].size() << ", addRemove=" << addRemovePt << endl;
    goodFeaturesToTrack(myFData.origGray,       // gray scale image.
                    myFData.origFeatures,      // List of points to track.
                    myFData.maxPoints,             // Max points to track.
                    0.01,           // quality level.
                    10,             // min distance between points.
                    Mat(),          // mask.
                    3,              // block size.
                    3,              // gradient size.
                    0,              // useHarrisDetector = false
                    0.04);          // free parameter for harris detector.

    //cout << "size 1: " << features.size() << endl;
    cornerSubPix(myFData.origGray, myFData.origFeatures, myFData.subPixWinSize, Size(-1,-1), myFData.termcrit);
}


// Figure out how much camera has moved, from the reference position. Return displacement in xoffset, yoffset.
// Calculate optical flow from original image to this one, take matched features, sort, look at 50 with smallest
// displacement (using smallest on assumption that biggest ones are mismatches or due to objects moving.)
// Calculate average x and y displacements for those 50 features, return results.
void getCameraJiggleOffset(Mat img, FlowData &myFData, int &xoffset, int &yoffset){

    cvtColor(img, myFData.newGray, COLOR_BGR2GRAY);

    calcOpticalFlowPyrLK(myFData.origGray,         // Image original features came from.
                            myFData.newGray,       // New image to look for features in.
                            myFData.origFeatures,  // Vector of original features.
                            myFData.newFeatures,   // Vector to put locations of features in new image.
                            myFData.status,        // Vector containing 1 if corresponding feature was found in new image.
                            myFData.err,           // Vector containing error amounts for corresponding features.
                            myFData.winSize,       // Size of search window at each pyramid level
                            3,                      // 0-based maximum pyramid levels to use.
                            myFData.termcrit,      // Termination criteria.
                            0,                      // flags.
                            0.001                   // Used to remove features that don't stand out.
    );

    // Get new list of points, sorted by total displacement - i.e. Dx^2 + Dy^2. Sort by that, take smallest X points.
    // Calculate average X and Y displacements for each of those points.
    // Step 1, calculate displacement proxy for each point that was found.
    vector<float> moves;
    vector<int> xdiffs, ydiffs;
    int xdiff, ydiff;
    int featNotFound = 0;       // Count features not found in new frame.
    for (int i = 0; i < myFData.origFeatures.size(); ++i) {
        // Don't look at features that weren't found.
        if (myFData.status[i]) {
            xdiff = myFData.newFeatures[i].x - myFData.origFeatures[i].x;
            ydiff = myFData.newFeatures[i].y - myFData.origFeatures[i].y;
            xdiffs.push_back(xdiff);
            ydiffs.push_back(ydiff);
            float tmove = pow(xdiff, 2) + pow(ydiff, 2);
            moves.push_back(tmove);
        } else { ++featNotFound; }
    }
    
    // Step 2, sort them, smallest first.
    Mat index;
    sortIdx(moves, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
    
    // Step 3, sum x and y displacements on first elements, then divide to get average.
    int numPts = 50;
    if( moves.size() < 50) numPts = moves.size();
    float xsum = 0, ysum = 0;
    cout << "features not found: " << featNotFound << " numPts: " << numPts << endl;
    int calcIndex;
    for (int i = 0; i < numPts; ++i) {
        calcIndex = index.at<int>(i);
        xsum += xdiffs[calcIndex];
        ysum += ydiffs[calcIndex];
    }
    xoffset = (int) (xsum / numPts);
    yoffset = (int) (ysum / numPts);
}


// Add green circle for each found feature, red for each not found, to given image.
void addCircles(Mat &camImg, FlowData &myFData) {
    
    Scalar green(0,255,0);
    Scalar red(0,0,255);
    Scalar myColor;
    for (int i = 0; i < myFData.newFeatures.size(); ++i) {
        if (myFData.status[i]) {myColor = green;}
        else {myColor = red;}
        circle(camImg, Point(myFData.newFeatures[i]), 5, myColor, 2);
    }
}

// Puts origImg into destImg, offset by given amounts. Assumes destImg is big enough to fit image plus offset.
void offsetImg(const Mat &origImg, Mat &destImg, int xoffset, int yoffset) {
    origImg.copyTo(destImg(Rect(xoffset, yoffset, origImg.cols, origImg.rows)));
}



// Use OpenCV sample program lkdemo.cpp as the basis for a program to perform simple image stabilization.
// Display stabilized results in the center of a larger window, so that the frame may wander while the 
// first points remain stable.
// Receives: Name of video file, O (capital letter o) if want to display original video, no processing.
// Design: Am going to use one of my traffic videos where the camera was handheld, and thus it moved a bit - 
// traffic1.MOV and traffic2.MOV.
// Will automatically initialize points to follow on first frame. Find 50 best matched points on subsequent frames,
// find average displacement, move all pixels by that amount. In phase 2, may reset points every 100 frames,
// to handle cases where camera is intentionally moving.
// To properly display, create a fake image to start, all zeros, copy real image into it, at offset of 50, 50, for example.
void exer16_9(int argc, char **argv) {
//void exer16_9(string videoName, string opt) {

    string videoName = argv[1];
    string opt = "";
    if (argc > 2) {
        opt = argv[2];
    }
    
    VideoCapture cap;
    cap.open(videoName);

    Mat displayImg, camImg;
    int bigWid, bigHght;
    int borderSize = 50;
    bigWid  = cap.get(VideoCaptureProperties::CAP_PROP_FRAME_WIDTH) + borderSize * 2;
    bigHght = cap.get(VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT) + borderSize * 2;
    int xoffset = borderSize, yoffset = borderSize;
    FlowData myFData;
    displayImg = Mat::zeros(bigHght, bigWid, CV_8UC3);

    //for (int frameCtr = 0; frameCtr < 15; ++frameCtr) { // For troubleshooting/testing.
    for (int frameCtr = 0; ; ++frameCtr) {
        cap >> camImg;
        if (camImg.empty()) exit;
        
        if ( opt != "O") {  // If want to display original video, don't to any proc.
            if (frameCtr == 0) {
                getFeatures(camImg, myFData);
            }

            // Calculate amount to move image by. First, figure out how much this image has moved,
            // compared to the reference. Then, add borderSize to it, so that, if it didn't move, it
            // will be centred in the output window. If offset will put us outside our display window, 
            // cap it so we stay inside.
            getCameraJiggleOffset(camImg, myFData, xoffset, yoffset);
            cout << "before offset: " << xoffset << ", " << yoffset;
            xoffset = borderSize - xoffset;
            yoffset = borderSize - yoffset;
            //cout << "offset: " << xoffset << ", " << yoffset << endl;
            if (xoffset > 2 * borderSize) {
                xoffset = 2 * borderSize;
            } else if (xoffset < 0) {
                xoffset = 0;
            }
            if (yoffset > 2 * borderSize) {
                yoffset = 2 * borderSize;
            } else if (yoffset < 0) {
                yoffset = 0;
            }

            cout << " after offset: " << xoffset << ", " << yoffset << endl;

            displayImg = Mat::zeros(bigHght, bigWid, CV_8UC3);
            //describe_mat(camImg, "camImg");
            //describe_mat(displayImg, "displayImg");
            addCircles(camImg, myFData);
        }
        offsetImg(camImg, displayImg, xoffset, yoffset);
        imshow("Stabilized", displayImg);
        waitKey(40);
    }
    cap.release();
    
    return;
}

