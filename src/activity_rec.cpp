#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <tf/transform_listener.h>
#include <geometry_msgs/Twist.h>
//#include <vector>
#include <fstream>
#include "hCRF.h"

#include <sstream>
#include "GraphUtils.h"
#include <human_action_recognition/activityRecognition.h>

using namespace std;



string filenameModel = "model.txt";
string filenameFeatures = "features.txt";
//string filenameOutput = "result.txt";
//string filenameStats = "stats.txt";

class activity_recognition{
public:
  ros::NodeHandle nh_;
  ros::Publisher action_result_pub;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub;
  image_transport::Publisher image_pub;
  int frame_length;
  int sk_feature_num;
  int current_frame;  
  vector<float> result;
  vector<vector<float> > visual_result;
  vector<string> label_name;
  float gaussian_mask[5];
  int nblabel;
  int counter;
  dMatrix* sk_data;
  DataSequence* data_seq;
  Toolbox* toolbox;
  Model* pModel;
  InferenceEngine* pInfEngine;
  

  activity_recognition(int frame_len,int sk_num)
    :it_(nh_),
     frame_length(frame_len),
     sk_feature_num(sk_num),
     counter(0),
     gaussian_mask{0.054488685,0.24420135,0.40261996,0.24420135,0.054488685}
  {
    current_frame = 0;    
    sk_data = new dMatrix(frame_length,sk_feature_num,0);
    data_seq = new DataSequence;
    toolbox = new ToolboxHCRF(3,OPTIMIZER_BFGS,0);
    toolbox->load((char*)filenameModel.c_str(),(char*)filenameFeatures.c_str());
    pModel=toolbox->getModel();
    pInfEngine=toolbox->getInferenceEngine();
    nblabel=pModel->getNumberOfSequenceLabels();
    result.resize(nblabel);
    visual_result.resize(nblabel);
    string label[4]={"drinking","shake hand","wave hand","natural"};
    label_name.assign(label, label+4);
    image_sub = it_.subscribe("/kinect_head_c2/rgb/image_rect_color",1,&activity_recognition::imageCb, this);
    image_pub = it_.advertise("/human_action/image_with_graph",10);

    //    action_result_pub=nh_.advertise<human_action_recognition::activityRecognition>("human_action_rec_result",1);
 }

  ~activity_recognition()
  {
    if(toolbox)
      {delete toolbox;
	toolbox = NULL;
      }
     if(data_seq)
      {
       	delete data_seq;
	data_seq = NULL;
	sk_data = NULL;
      }
   
    
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
      {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
      }
    catch (cv_bridge::Exception& e)
      {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
      }
    IplImage recevied_img = cv_ptr->image;
    IplImage *display_img = &recevied_img; 

    if(visual_result[0].size()>=frame_length)
      {
        for(int i=0; i<nblabel; i++)
          drawFloatGraph(&visual_result[i][0], visual_result[i].size()-2 ,display_img, 0, 1, 400, 200, label_name[i].c_str());//output the graph have been smoothed thus -2                                
        showImage(display_img,10);
        setGraphColor(0);
      }
    image_pub.publish(cv_ptr->toImageMsg());
  }


  double v_norm(vector<double> &v)
  {
    double sum=0.0;
    for(int i=0; i<v.size(); i++){
      sum+=v[i]*v[i];
    }
    return sqrt(sum);
  }
  vector<double> v_sub(tf::StampedTransform transform1, tf::StampedTransform transform2)
  {
    vector<double> v;
    v.push_back(transform1.getOrigin().x()-transform2.getOrigin().x());
    v.push_back(transform1.getOrigin().y()-transform2.getOrigin().y());
    v.push_back(transform1.getOrigin().z()-transform2.getOrigin().z());
    return v;
  }  

  float gaussian_smooth(vector<float> v) //since its a sliding, we only need to smooth the third to last element;  
  {
    float value;
    vector<float>::reverse_iterator iter=v.rbegin(); 
    if(v.size() < 5) 
      return 0;
    else
      {
	for(int i=0; i<5; i++)
	  {
	    value += *iter*gaussian_mask[i];
	    iter++;
	  }
	return value;
      }
  }


};



int main(int argc, char** argv)
{
  int sk_feature_num = 27;
  int frame_length = 30;
  ros::init(argc, argv, "pose_descriptor_extractor");
  activity_recognition ar(frame_length,sk_feature_num);
  ros::Rate rate(10.0);
  tf::TransformListener listener;
  tf::StampedTransform transform;
  vector<tf::StampedTransform> transform_v; 
  bool new_msg_flat=false;
  transform_v.resize(9);
  string frame_name[9]={"/head_1","/neck_1","/torso_1","/left_shoulder_1","/left_elbow_1","/left_hand_1","/right_shoulder_1","/right_elbow_1","/right_hand_1"};

  while(ar.nh_.ok())
    {
      ros::spinOnce();
      try{
	for(int i=0 ; i<9; i++){
	  listener.lookupTransform("/head_mount_kinect_ir_link", frame_name[i] ,ros::Time(0) ,transform);

	  transform_v[i]=transform;
	}	

	new_msg_flat=true;
      }
      catch (tf::TransformException ex)
        {
	  
          ros::Duration(1.0).sleep();
        }

      if(new_msg_flat==true)
	{
	
	  vector<double> d_l = ar.v_sub(transform_v[5],transform_v[4]);
	  vector<double> d_r = ar.v_sub(transform_v[8],transform_v[7]);      
	  double ref_dis=max(ar.v_norm(d_l), ar.v_norm(d_r)); //use the ||left/right hand - elbow|| as reference distance
	  // cout<<"distance"<<ref_dis<<endl;
	  vector<vector<double> > p(transform_v.size());
	
	  for(int i=0; i<transform_v.size(); i++)
	    {

	      vector<double> temp=(ar.v_sub(transform_v[i],transform_v[2])); // every joint relative to head
	      for(int j=0; j<temp.size(); j++)
		{
		  p[i].push_back(temp[j]/ref_dis);
		
		  if(ar.current_frame < ar.frame_length-1)
		    ar.sk_data->setValue(i*temp.size()+j,ar.current_frame,p[i][j]); //insert one frame (24 points)
		  if(ar.current_frame == ar.frame_length-1)
		    {
		      ar.sk_data->setValue(i*temp.size()+j,ar.current_frame,p[i][j]);  
		    }

		}
	    }
	  
	  if(ar.current_frame < ar.frame_length-1)
	    ar.current_frame++;
	  if(ar.current_frame == ar.frame_length-1)
	    ar.sk_data->slid_window(frame_length,sk_feature_num);
	    
	  ar.data_seq->setPrecomputedFeatures(ar.sk_data);


	  // cout<<"result "<<ar.toolbox->realtimeLabelOutput(ar.data_seq)<<endl;
	  //int label = ar.toolbox->realtimeLabelOutput(ar.data_seq);
	  stringstream ss;
	  human_action_recognition::activityRecognition action_result;
	  //ar.result[label]++;
	 
	  dMatrix* score;
	  int score_sum=0;
	  score = ar.toolbox->realtimeScoreOutput(ar.data_seq);
	 
	  for(int i=0; i<ar.nblabel; i++)
	    {
	      ar.result[i] += score->getValue(i,0);	      
	 
	    }
	  human_action_recognition::activityRecognition ar_msg;
	  //----------------------------------
	  int lowest_score = ar.result[0];
		  for(int i=1; i<ar.nblabel; i++)
		    {
		      if(ar.result[i] <= lowest_score)
			lowest_score = ar.result[i];		      
		    }
		  for(int i=0; i<ar.nblabel; i++)
		    {
		      ar.result[i] -= lowest_score;
		      score_sum += ar.result[i]; 
		    }
		  for(int i=0; i<ar.nblabel; i++)
		    {
		      ar.result[i] = ar.result[i]/score_sum;
		    
		      if(ar.visual_result[i].size()>=frame_length)
			{
			  ar.visual_result[i].erase(ar.visual_result[i].begin());
			  ar.visual_result[i][frame_length-3]=ar.gaussian_smooth(ar.visual_result[i]);
			  if(ar.gaussian_smooth(ar.visual_result[i]) > 0.75)
			    {
                              cout<<"gonna pub"<<ar.label_name[i]<<endl;
			      ar_msg.label=ar.label_name[i];
			      if(ar.label_name[i]=="shake hand")
				{
                                  geometry_msgs::Point interactionPoint;
				  if(transform_v[5].getOrigin().z()>transform_v[8].getOrigin().z())
				    {
				      interactionPoint.x = transform_v[5].getOrigin().x();
				      interactionPoint.y = transform_v[5].getOrigin().y();
				      interactionPoint.z = transform_v[5].getOrigin().z();
				    }
				  else
				    {
				      interactionPoint.x = transform_v[8].getOrigin().x();
				      interactionPoint.y = transform_v[8].getOrigin().y();
				      interactionPoint.z = transform_v[8].getOrigin().z();
				    }
				  ar_msg.interactionPoint=interactionPoint;
				}
                              //  ar.action_result_pub.publish(ar_msg);
			    }
			}
			  
		      ar.visual_result[i].push_back(ar.result[i]);
		      		
		    }
		  
                  /* IplImage* graph=drawFloatGraph(&ar.visual_result[0][0], ar.visual_result[0].size()-2,NULL, 0, 1, 400, 200, ar.label_name[0].c_str());
		  if(ar.visual_result[0].size()>=frame_length)
		    {
		      for(int i=1; i<ar.nblabel; i++)
			drawFloatGraph(&ar.visual_result[i][0], ar.visual_result[i].size()-2 ,graph, 0, 1, 400, 200, ar.label_name[i].c_str());//output the graph have been smoothed thus -2
		      showImage(graph,10);
		      setGraphColor(0);
                      }*/
	  	      
                  // listener.clear();
		  
	}
      new_msg_flat=false;
      rate.sleep();

    }
 return 0;
}
