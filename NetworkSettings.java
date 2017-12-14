package stocks.marketanalysis.Neuralnetwork.neuralnetwork;

import stocks.marketanalysis.daos.IGeneralizedDao;
import stocks.marketanalysis.daos.Impl.NetworkConfigurationDaoImpl;
import stocks.marketanalysis.models.NetworkConfiguration;

/**
 *
 * @author RajaramPakur
 */
public class NetworkSettings {
    
        private double momentum;

	private double learningRate;
        
        private double threshold;

	private int epochCount;

	private int numberLayer;

	private int[] layerSize;


    public NetworkSettings(){
    }
    
    /*
    This constructor is used to initialize the configuration value for neural network
    @param momentum, learningRate, epochCount, numberLayer, layerSize
    */     
    public NetworkSettings(double momentum,double learningRate, int epochCount, int numberLayer, int[] layerSize, double threshold){
        
        this.momentum = momentum;
        this.learningRate = learningRate;
        this.epochCount = epochCount;
        this.numberLayer = numberLayer;
        this.layerSize = layerSize;
        this.threshold = threshold;
        
        NetworkConfiguration objConfig = new NetworkConfiguration();
        objConfig.setIteration(epochCount);
        objConfig.setMomentum(momentum);
        objConfig.setLayer(numberLayer);
        String s = "";
        for(Integer l: layerSize){
             s = s+l+",";
        }
        objConfig.setLayersize(s);
        objConfig.setLearningrate(learningRate);
        objConfig.setThreshold(threshold);
        System.out.println(objConfig.getIteration());
        IGeneralizedDao<NetworkConfiguration> objNetCon =new NetworkConfigurationDaoImpl();
        objNetCon.insert(objConfig);
        
        
        
    }

    /**
     * @return the momentum
     */
    public double getMomentum() {
        return momentum;
    }

    /**
     * @param momentum the momentum to set
     */
    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }

    /**
     * @return the learningRate
     */
    public double getLearningRate() {
        return learningRate;
    }

    /**
     * @param learningRate the learningRate to set
     */
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    /**
     * @return the epochCount
     */
    public int getEpochCount() {
        return epochCount;
    }

    /**
     * @param epochCount the epochCount to set
     */
    public void setEpochCount(int epochCount) {
        this.epochCount = epochCount;
    }

    /**
     * @return the numberLayer
     */
    public int getNumberLayer() {
        return numberLayer;
    }

    /**
     * @param numberLayer the numberLayer to set
     */
    public void setNumberLayer(int numberLayer) {
        this.numberLayer = numberLayer;
    }

    /**
     * @return the layerSize
     */
    public int[] getLayerSize() {
        return layerSize;
    }

    /**
     * @param layerSize the layerSize to set
     */
    public void setLayerSize(int[] layerSize) {
        this.setLayerSize(layerSize);
    }

    /**
     * @return the threshold
     */
    public double getThreshold() {
        return threshold;
    }

    /**
     * @param threshold the threshold to set
     */
    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }

    
    
    
}
