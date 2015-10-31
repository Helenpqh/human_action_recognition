#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <geometry_msgs/Twist.h>
#include <vector>
#include <fstream>
#include "hCRF.h"

using namespace std;



string filenameModel = "model.txt";
string filenameFeatures = "features.txt";
string filenameOutput = "result.txt";
string filenameStats = "stats.txt";

class activity_recognition{
public:
  ros::NodeHandle nh_;
  int frame_length;
  int sk_feature_num;
  int current_frame;  
  int result[3];//number of labels;
  int counter;
  dMatrix* sk_data;
  DataSequence* data_seq;
  DataSet* data;// = new DataSet;
  ToolboxHCRF* toolbox;
  
  
  activity_recognition(int frame_len,int sk_num)
    :frame_length(frame_len),sk_feature_num(sk_num),counter(0)
  {
    for(int i=0; i<3; i++)
      {
	result[i]=0;
      }
    current_frame = 0;    
    //    sk_feature_num  = 24; //3*8 without head
    sk_data = new dMatrix(frame_length,sk_feature_num,0);
    data_seq = new DataSequence;
    data = new DataSet;
    toolbox = new ToolboxHCRF(3,OPTIMIZER_BFGS,0);
    toolbox->load((char*)filenameModel.c_str(),(char*)filenameFeatures.c_str());
  }

  ~activity_recognition()
  {
    if(toolbox)
      {delete toolbox;
	toolbox = NULL;
      }
     if(data)
      {
       	delete data;
	data = NULL;
	data_seq = NULL;
	sk_data = NULL;
      }
   
    
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




};



int main(int argc, char** argv)
{
  int sk_feature_num = 24;
  int frame_length = 50;
  ros::init(argc, argv, "pose_descriptor_extractor");
  activity_recognition ar(frame_length,sk_feature_num);
  ros::Rate rate(10.0);
  tf::TransformListener listener;

  /*  fstream fout;
  fout.open(file_name, ios::app|ios::out);
  if(!fout.is_open()){
    cerr << "cannnot open file" << endl;
    return false ;
    }*/

      tf::StampedTransform transform;
      vector<tf::StampedTransform> transform_v;      
  
  while(ar.nh_.ok())
    {

      try{

      listener.lookupTransform("/openni_depth_frame", "/head_1",ros::Time(0) ,transform);
      transform_v.push_back(transform);
      listener.lookupTransform("/openni_depth_frame", "/neck_1",ros::Time(0) ,transform);
      transform_v.push_back(transform);
      listener.lookupTransform("/openni_depth_frame","/torso_1", ros::Time(0) ,transform);
      transform_v.push_back(transform);

      listener.lookupTransform("/openni_depth_frame","/left_shoulder_1", ros::Time(0) ,transform);
      transform_v.push_back(transform);
      listener.lookupTransform("/openni_depth_frame", "/left_elbow_1",ros::Time(0) ,transform);
      transform_v.push_back(transform);
      listener.lookupTransform("/openni_depth_frame","/left_hand_1", ros::Time(0) ,transform);
      transform_v.push_back(transform);

      listener.lookupTransform("/openni_depth_frame","/right_shoulder_1", ros::Time(0) ,transform);
      transform_v.push_back(transform);
      listener.lookupTransform("/openni_depth_frame","/right_elbow_1", ros::Time(0) ,transform);
      transform_v.push_back(transform);
      listener.lookupTransform("/openni_depth_frame","/right_hand_1", ros::Time(0) ,transform);
      transform_v.push_back(transform);
      
      /*      listener.lookupTransform("/openni_depth_frame","/left_hip_1", ros::Time(0) ,transform);
      transform_v.push_back(transform);
      listener.lookupTransform("/openni_depth_frame","/left_knee_1", ros::Time(0) ,transform);
      transform_v.push_back(transform);
      listener.lookupTransform("/openni_depth_frame","/left_foot_1", ros::Time(0) ,transform);
      transform_v.push_back(transform);

      listener.lookupTransform("/openni_depth_frame","/right_hip_1", ros::Time(0) ,transform);
      transform_v.push_back(transform);
      listener.lookupTransform("/openni_depth_frame","/right_knee_1", ros::Time(0) ,transform);
      transform_v.push_back(transform);
      listener.lookupTransform("/openni_depth_frame","/right_foot_1", ros::Time(0) ,transform);
      transform_v.push_back(transform);*/
      }
      catch (tf::TransformException ex)
        {
          ros::Duration(1.0).sleep();
        }
      if(!transform_v.empty())
	{
	
	  vector<double> d_l = ar.v_sub(transform_v[5],transform_v[4]);
	  vector<double> d_r = ar.v_sub(transform_v[8],transform_v[7]);      
	  double ref_dis=max(ar.v_norm(d_l), ar.v_norm(d_r)); //use the ||left/right hand - elbow|| as reference distance
	  //        cout<<"distance"<<ref_dis<<endl;
	  vector<vector<double> > p(transform_v.size()-1);
	
	  for(int i=0; i<transform_v.size()-1; i++)
	    {

	      vector<double> temp=(ar.v_sub(transform_v[i+1],transform_v[0])); // every joint relative to head
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
	  
	  ar.data_seq->setPrecomputedFeatures(ar.sk_data);
	  cout<<"result "<<ar.toolbox->realtimeOutput(ar.data_seq)<<endl;
	  switch (ar.toolbox->realtimeOutput(ar.data_seq))
		{
		case 0:
		  ar.result[0]++;
		  break;
		case 1:
		  ar.result[1]++;
		  break;
		case 2:
		  ar.result[2]++;
		  break;
		default:
		  break;
		}	
	      if(ar.counter>=10)
		{
		 
		  cout<<"working: "<<ar.result[0]<<" shake hand: "<<ar.result[1]<<" drinking: "<<ar.result[2]<<endl;
		  ar.counter=0;
		  for(int i=0; i<3; i++)
		    {
		     ar.result[i]=0;
		    }
		}
		ar.counter++;
	    
	}
      //      listener.clear();
      transform_v.clear();
      rate.sleep();

    }

return 0;
}


