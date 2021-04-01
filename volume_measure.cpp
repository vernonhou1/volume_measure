#include <iostream>                         //�����������ͷ�ļ�
#include <vector>

#include <pcl/io/real_sense_2_grabber.h>         //OpenNI�ɼ�ͷ�ļ�

#include <pcl/ModelCoefficients.h>          //����һ����ģ�������ͷ�ļ�

#include <pcl/point_types.h> 
#include <pcl/pcl_base.h>               //�������� ������ͷ�ļ�

#include <pcl/visualization/cloud_viewer.h> //���ӻ�ͷ�ļ�

#include <pcl/filters/passthrough.h>                    //ֱͨ�˲���ͷ�ļ�
#include <pcl/filters/voxel_grid.h>                     //�����˲���ͷ�ļ�
#include <pcl/filters/extract_indices.h>                //������ȡ�˲���ͷ�ļ�
#include <pcl/filters/statistical_outlier_removal.h>    //ͳ���˲�ͷ�ļ�
#include <pcl/filters/project_inliers.h>                //ӳ�����ͷ�ļ�

#include <pcl/sample_consensus/method_types.h>  //����������Ʒ���ͷ�ļ�
#include <pcl/sample_consensus/model_types.h>   //ģ�Ͷ���ͷ�ļ�

#include <pcl/segmentation/sac_segmentation.h>  //���ڲ���һ���Էָ�����ͷ�ļ�

#include <pcl/surface/concave_hull.h>           //��ȡ͹����������ε�ͷ�ļ�

//#include <opencv2/opencv.hpp>   //opencvͷ�ļ�

using namespace std;
//using namespace cv;

class SimpleOpenNIProcessor
{
public:
	pcl::visualization::CloudViewer viewer; // Cloudviewer��ʾ����

											// ֱͨ�˲�������
	pcl::PassThrough<pcl::PointXYZ> XpassFilter;
	pcl::PassThrough<pcl::PointXYZ> YpassFilter;

	// �²��� VoxelGrid �˲�����
	pcl::VoxelGrid<pcl::PointXYZ> VoxlFilter;

	// �����ָ����
	pcl::SACSegmentation<pcl::PointXYZ> segPlane;

	// ��ȡ�����˲���
	pcl::ExtractIndices<pcl::PointXYZ> extractPlane;

	// ͳ���˲���
	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> statisFilter;

	// ����ͶӰ�˲�����
	pcl::ProjectInliers<pcl::PointXYZ> projec;
	int heightCount = 0;
	float heightSum = 0.0;

	int areaCount = 0;
	float areaSum = 0.0;


	SimpleOpenNIProcessor() : viewer("PCL OpenNI Viewer") {
		// �˲��ֶ���Ϊx�᷽��
		XpassFilter.setFilterFieldName("x");
		// �趨�ɽ��ܷ�Χ�������ڷ�Χ�ڵĵ���˵����߱�������setFilterLimitsNegative������
		XpassFilter.setFilterLimits(-0.25, 0.05);  // TODO: X����ü���ֵ��Ҫ���� ʵ����0.44m

												   // �˲��ֶ���Ϊy�᷽��
		YpassFilter.setFilterFieldName("y");
		// �趨�ɽ��ܷ�Χ�������ڷ�Χ�ڵĵ���˵����߱�������setFilterLimitsNegative������
		YpassFilter.setFilterLimits(-0.27, 0.18);  // TODO: Y����ü���ֵ��Ҫ���� ʵ����0.69m

												   // ���������˲�ʱ�������������Ϊ0.5*0.5*0.5cm��������
		VoxlFilter.setLeafSize(0.005f, 0.005f, 0.005f);

		// ͳ���˲���ʼ����
		statisFilter.setMeanK(50);  // ��ÿ����������ٽ��������Ϊ50
		statisFilter.setStddevMulThresh(1.0);   // ����׼�����Ϊ1����ζ��һ����ľ��볬��ƽ������1����׼�����ϣ��ͻᱻ���Ϊ��Ⱥ�㣬�����Ƴ���

												// ��ѡ������ģ��ϵ����Ҫ�Ż�
		segPlane.setOptimizeCoefficients(true);
		// ��ѡ�����÷ָ��ģ�����͡����õ�����������Ʒ�����������ֵ��������������
		segPlane.setModelType(pcl::SACMODEL_PLANE);
		segPlane.setMethodType(pcl::SAC_RANSAC);
		segPlane.setDistanceThreshold(0.01);    // TODO: ������ֵ��Ҫ����
		segPlane.setMaxIterations(1000);

		// ����ͶӰģ��ΪSACMODEL_PLANE
		projec.setModelType(pcl::SACMODEL_PLANE);
	}  // Construct a cloud viewer, with a window name

  // ����ص�����cloud_cb_,��ȡ������ʱ�����ݽ��д���
	void cloud_cb_(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud)
	{
		heightCount++;
		areaCount++;

		cout << "ԭʼ���ƴ�С��" << cloud->points.size() << endl;

		// 1. �ռ�ü�
		pcl::PointCloud<pcl::PointXYZ>::Ptr pass_filtered(new pcl::PointCloud<pcl::PointXYZ>);

		XpassFilter.setInputCloud(cloud);
		XpassFilter.filter(*pass_filtered);

		YpassFilter.setInputCloud(pass_filtered);
		YpassFilter.filter(*pass_filtered);

		cout << "�ռ�ü�����ƴ�С��" << pass_filtered->points.size() << endl;

		// 2. �²�����������ѹ��
		// �²���
		pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_filtered(new pcl::PointCloud<pcl::PointXYZ>);

		VoxlFilter.setInputCloud(pass_filtered);
		VoxlFilter.filter(*voxel_filtered);

		cout << "�²�������ƴ�С��" << voxel_filtered->points.size() << endl;

		// 3. TODO: ����һ���˲���ɾ���쳣��
		// 3. ͳ���˲� ȥ���쳣��
		pcl::PointCloud<pcl::PointXYZ>::Ptr statisFiltered(new pcl::PointCloud<pcl::PointXYZ>);
		statisFilter.setInputCloud(pass_filtered);
		statisFilter.filter(*statisFiltered);

		// 3. ƽ��ָ� ��ȡ����
		// �����ָ�ʱ����Ҫ��ģ��ϵ������ coefficientsPlane
		// �����洢ģ���ڵ�ĵ��������϶��� inliersPlane
		pcl::ModelCoefficients::Ptr coefficientsPlane(new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr inliersPlane(new pcl::PointIndices());

		segPlane.setInputCloud(statisFiltered);
		segPlane.segment(*inliersPlane, *coefficientsPlane);
		if (inliersPlane->indices.size() == 0) {
			cout << "ƽ��ָ�ʧ�ܣ�û���ҵ�ƽ��" << endl;
		}
		//        cout << "ƽ��ָ����ƽ��������ĿΪ" << inliersPlane->indices.size() << endl;
		//        cout << "ƽ��ָ����ƽ��ģ�ͣ�ax+by+cz+d=0��ʽ������" << ", a = " << coefficientsPlane->values[0]
		//             << ", b = " << coefficientsPlane->values[1]
		//             << ", c = " << coefficientsPlane->values[2]
		//             << ", d = " << coefficientsPlane->values[3] << endl;

		// 3.1 ����ƽ����ɫ
		//        // 3.1.1 ����һ��ƽ����ƣ�������Ϊ�ײ�������ȡ������
		//        pcl::PointCloud<pcl::PointXYZ>::Ptr PlaneCloud(new pcl::PointCloud<pcl::PointXYZ>);
		//        pcl::copyPointCloud(*voxel_filtered, inliersPlane->indices, *PlaneCloud);

		// 3.1.2 ����һ��RGB���ƣ����Ա����ɫ
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr ColorCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
		pcl::copyPointCloud(*statisFiltered, *ColorCloud);

		// 3.1.3 �Եײ�ƽ���ڵ������ɫ��� // TODO: ����Ҫ�Ļ����������ｫindcies����һ��
		for (int i = 0; i < inliersPlane->indices.size(); ++i) {
			ColorCloud->points[inliersPlane->indices[i]].r = 255;
			ColorCloud->points[inliersPlane->indices[i]].g = 0;
			ColorCloud->points[inliersPlane->indices[i]].b = 0;
		}

		// 4. ����������ȡ������֮��ĵ���
		pcl::PointCloud<pcl::PointXYZ>::Ptr extracted(new pcl::PointCloud<pcl::PointXYZ>);
		extractPlane.setInputCloud(statisFiltered);
		extractPlane.setIndices(inliersPlane);
		extractPlane.setNegative(true); // ����Negative�ǻ�ȡ����ƽ��ĵ�
		extractPlane.filter(*extracted);
		cout << "�����˲�����ȥ�������ƴ�С" << extracted->points.size() << endl;

		// 5. ƽ��ָ� ��ȡĿ�����ϱ���
		pcl::ModelCoefficients::Ptr coefficientsTarget(new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr inliersTarget(new pcl::PointIndices());

		segPlane.setInputCloud(extracted);
		segPlane.segment(*inliersTarget, *coefficientsTarget);
		if (inliersPlane->indices.size() == 0) {
			cout << "ƽ��ָ�ʧ�ܣ�û���ҵ�ƽ��" << endl;
			return;
		}
		//        cout << "ƽ��ָĿ�궥��������ĿΪ" << inliersTarget->indices.size() << endl;
		//        cout << "ƽ��ָĿ�궥��ƽ��ģ�ͣ�ax+by+cz+d=0��ʽ������" << ", a = " << coefficientsTarget->values[0]
		//             << ", b = " << coefficientsTarget->values[1]
		//             << ", c = " << coefficientsTarget->values[2]
		//             << ", d = " << coefficientsTarget->values[3] << endl;


		// TODO: �˴����Լ�������ĸ߶�
		//        float height = abs(coefficientsPlane->values[3] - coefficientsTarget->values[3]);
		// �Ľ�����
		float height = abs(coefficientsPlane->values[3] / coefficientsPlane->values[2] -
			coefficientsTarget->values[3] / coefficientsPlane->values[2]);
		heightSum += height;
		float avheight = heightSum / heightCount;
		cout << "����ĸ߶�Ϊ height = " << avheight << "m" << endl;

		// 5.1 ����Ŀ�궥����ɫ
		//        // 5.1.1 ����һ��Ŀ�궥����ƣ�������Ϊ�ײ�������ȡ������
		//        pcl::PointCloud<pcl::PointXYZ>::Ptr TargetCloud(new pcl::PointCloud<pcl::PointXYZ>);
		//        pcl::copyPointCloud(*extracted, inliersTarget->indices, *TargetCloud);

		// 5.1.2 ������ɫ
		for (int i = 0; i < inliersTarget->indices.size(); ++i) {
			ColorCloud->points[inliersTarget->indices[i]].r = 0;
			ColorCloud->points[inliersTarget->indices[i]].g = 255;
			ColorCloud->points[inliersTarget->indices[i]].b = 0;
		}

		// 6. ͶӰƽ��ģ��ϵ��
		pcl::PointCloud<pcl::PointXYZ>::Ptr target_projected(new pcl::PointCloud<pcl::PointXYZ>);

		projec.setIndices(inliersTarget);
		projec.setInputCloud(statisFiltered);
		projec.setModelCoefficients(coefficientsTarget);
		projec.filter(*target_projected);
		cout << "ͶӰ��ͶӰ�����ĿΪ" << target_projected->points.size() << endl;

		// 7. ΪͶӰ�ĵ㴴����������
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::ConcaveHull<pcl::PointXYZ> chull;

		//polygons�洢һϵ�ж��㼯��ÿ�鶥�㼯��ʾһ������� //Ӧ��ȡ���е�0�����㼯
		std::vector<pcl::Vertices> polygons;
		chull.setInputCloud(target_projected);
		chull.setAlpha(0.1);

		// polygons��Ӧ����ǰ��point cloud_hull������
		chull.reconstruct(*cloud_hull, polygons);

		cout << "������ĿΪ" << cloud_hull->points.size() << endl;
		cout << "��⵽�İ�����ĿΪ" << polygons.size() << endl;
		cout << "��⵽�İ����㼯��ĿΪ" << polygons[0].vertices.size() << endl;

		//       //  7.1 ��������㼯�鿴
		//        for (int j = 0; j < cloud_hull->points.size(); ++j) {
		//            cout << cloud_hull->points[j].x << " " <<  cloud_hull->points[j].y << " " <<  cloud_hull->points[j].z << endl;
		//        }
		// 7.1.2 ��ȡ��ά������OpenCV����

		// ���ɶ�ά����
		//std::vector<cv::Point2f> hullContour;
		//for (int j = 0; j < polygons[0].vertices.size(); ++j) {
		//	hullContour.push_back(cv::Point2f(cloud_hull->points[polygons[0].vertices[j]].x,
		//		cloud_hull->points[polygons[0].vertices[j]].y));
		//}

		//cv::RotatedRect minRect = cv::minAreaRect(hullContour);

		//// ����ĳ���������
		//cout << "��С��Ӿ��εĳ��� " << minRect.size.width << " ���� " << minRect.size.height << " ����� " << minRect.size.area()
		//	<< " �Ƕȣ� " << minRect.angle << endl;


		//         7.2 ���ݶ��㼯���ڲ�ɫ�����б������ TODO: ���⣺����ĵ㼯���ɫ�ռ��еĵ㼯��Ӧ����

		//        pcl::PointCloud<pcl::PointXYZRGB>::Ptr ColorHull(new pcl::PointCloud<pcl::PointXYZRGB>);
		//        pcl::copyPointCloud(*cloud_hull, *ColorHull);
		//        for (int j = 0; j < polygons[0].vertices.size(); ++j) {
		//            ColorHull->points[polygons[0].vertices[j]].r = 0;
		//            ColorHull->points[polygons[0].vertices[j]].g = 0;
		//            ColorHull->points[polygons[0].vertices[j]].b = 255;
		//        }

		// 8. ���㰼�����
		// 8.1 �Դ���������
		// ����ļ��㷽ʽ��û������ģ�������ܳ����ϱ���ļ����
		float area = pcl::calculatePolygonArea(*cloud_hull);

		// 8.2��ѧԭ������
		//        float area = 0.0;
		//        float prevX = cloud_hull->points[0].x;
		//        float prevY = cloud_hull->points[0].y;
		//        for (int j = 0; j < cloud_hull->points.size(); ++j) {
		//            float nowX = cloud_hull->points[j].x;
		//            float nowY = cloud_hull->points[j].y;
		//            area += prevX*nowY - prevY*nowX;
		//            prevX = nowX;
		//            prevY = nowY;
		//        }
		//        area = area * 0.5;

		cout << "��ֵ���Ϊ " << area << endl;
		float realArea = area / (0.3 * 0.45) * (0.69 * 0.43);

		areaSum += realArea;
		float avarea = areaSum / areaCount;

		cout << "ʵ�����Ϊ " << avarea << "ƽ����" << endl;

		float realVolum = height * avarea;
		cout << "ʵ�����Ϊ " << realVolum << "������" << endl << endl << endl;


		if (!viewer.wasStopped()) // Check if the gui was quit. true if the user signaled the gui to stop
			viewer.showCloud(ColorCloud);
	}

  void run ()
  {
	  // �½�OpenNI�豸�Ĳ����� grabber
	  pcl::Grabber* interface = new pcl::RealSense2Grabber("1111");

	  // �����ص�
	  boost::function<void(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr&)> f =
		  boost::bind(&SimpleOpenNIProcessor::cloud_cb_, this, _1);

	  interface->registerCallback(f);

	  interface->start();

	  while (!viewer.wasStopped())
	  {
		  //boost::this_thread::sleep(boost::posix_time::milliseconds(10));
	  }

	  interface->stop();
  }

};

int main ()
{
  SimpleOpenNIProcessor v;
  v.run ();
  return (0);
}
