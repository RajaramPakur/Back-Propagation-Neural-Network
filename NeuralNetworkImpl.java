package stocks.marketanalysis.Neuralnetwork.neuralnetwork;

import java.util.List;
import org.hibernate.SQLQuery;
import org.hibernate.Session;
import stocks.marketanalysis.HibernateUtil;
import stocks.marketanalysis.Neuralnetwork.Activation.ActivationFunction;
import stocks.marketanalysis.Neuralnetwork.Activation.Sigmoid;
import stocks.marketanalysis.daos.IGeneralizedDao;
import stocks.marketanalysis.daos.Impl.NetworkConfigurationDaoImpl;
import stocks.marketanalysis.daos.Impl.TrainedDatasetDaoImpl;
import stocks.marketanalysis.models.NetworkConfiguration;
import stocks.marketanalysis.models.TrainedDataset;

/**
 *
 * @author RajaramPakur
 */
public class NeuralNetworkImpl implements NeuralNetwork {

    double[][] out;
    double[][] delta;
    double[][][] weight;
    double[][][] prevWeight;
    double learingRate;
    int config;
    double momentum;
    int numLayer;
    int[] layerSize;
    long id;
    NetworkConfiguration objConfig = new NetworkConfiguration();
    IGeneralizedDao<NetworkConfiguration> objNC = new NetworkConfigurationDaoImpl();
    ActivationFunction activationFunction = new Sigmoid();
    IGeneralizedDao<TrainedDataset> objTrain = new TrainedDatasetDaoImpl();
    TrainedDataset objData = new TrainedDataset();

    public NeuralNetworkImpl() {
    }

    /**
     *
     * @param numLayer
     * @param layerSize
     * @param learningRate
     * @param momentum
     * @param config
     */
    public NeuralNetworkImpl(int numLayer, int[] layerSize, double learningRate, double momentum, int config) {

        this.config = config;
        this.learingRate = learningRate;
        this.momentum = momentum;
        this.numLayer = numLayer;
        this.layerSize = new int[numLayer];

//       sets no of layers and their sizes
        for (int i = 0; i < this.numLayer; i++) {
            this.layerSize[i] = layerSize[i];
        }

//       allocate memory for output of each network
        this.out = new double[this.numLayer][];

        for (int i = 0; i < this.numLayer; i++) {
            out[i] = new double[this.layerSize[i]];
        }

//    allocate memory for weights
        this.delta = new double[this.numLayer][];
        this.weight = new double[this.numLayer][][];
        for (int i = 1; i < this.numLayer; i++) {
            this.weight[i] = new double[this.layerSize[i]][];
            this.delta[i] = new double[this.layerSize[i]];
        }

        for (int i = 1; i < this.numLayer; i++) {
            for (int j = 0; j < this.layerSize[i]; j++) {
                this.weight[i][j] = new double[this.layerSize[i - 1] + 1];
            }
        }

//  allocate memory for previouse weights
        this.prevWeight = new double[this.numLayer][][];
        for (int i = 1; i < this.numLayer; i++) {
            this.prevWeight[i] = new double[this.layerSize[i]][];
        }

        for (int i = 1; i < this.numLayer; i++) {
            for (int j = 0; j < this.layerSize[i]; j++) {
                this.prevWeight[i][j] = new double[this.layerSize[i - 1] + 1];
            }
        }

        objConfig.setLayer(numLayer);
        objConfig.setLearningrate(learningRate);
        objConfig.setMomentum(momentum);

        List<NetworkConfiguration> nList = objNC.getAll();
        for (NetworkConfiguration n : nList) {
            if (n.getLayer() == objConfig.getLayer() || n.getLearningrate() == objConfig.getLearningrate() || n.getMomentum() == objConfig.getMomentum()) {
                objConfig.setId(n.getId());
            }
        }

//     seed and assign random weights
        System.out.println("-------------Initial Random weight assigns-----------\n");

        for (int i = 1; i < this.numLayer; i++) {
            for (int j = 0; j < this.layerSize[i]; j++) {
                for (int k = 0; k < this.layerSize[i - 1] + 1; k++) {

                    if (config == 0) {
                        this.weight[i][j][k] = Math.random(); // assign the random value
                    } else {
                        if (k < this.layerSize[i - 1]) {
                            double kg = getWeight(i, j, k);
                            this.weight[i][j][k] = kg;
                        } else {
                            this.weight[i][j][k] = Math.random();
                        }

                    }
                    objData.setLayer(i);
                    objData.setLayernode(j);
                    objData.setInputnode(k);
                    objData.setWeight(weight[i][j][k]);
                    objData.setIteration(0);
                    objData.setOutputvalue(0.0);
                    objData.setObjconfig(objConfig);
                    objTrain.insert(objData);
                    System.out.println("[" + i + "][" + j + "][" + k + "][" + this.weight[i][j][k] + "]");
                }
            }
        }

//    initialize previous weights to 0 for first iteration
        for (int i = 1; i < this.numLayer; i++) {
            for (int j = 0; j < this.layerSize[i]; j++) {
                for (int k = 0; k < this.layerSize[i - 1] + 1; k++) {
                    this.prevWeight[i][j][k] = (double) 0.0; // assign the random value
                }
            }
        }

    }

    /**
     *
     * @param layer
     * @param layerNode
     * @param inputNode
     * @return double value weight of network
     */
    private double getWeight(int layer, int layerNode, int inputNode) {

        Double result = null;
        Session session = HibernateUtil.getSessionFactory().openSession();
        try {
            String string = "SELECT weight FROM traineddata WHERE config_id = '" + config + "' AND layer = '" + layer + "' AND layernode = '" + layerNode + "' AND inputnode = '" + inputNode + "' AND iteration = (SELECT MAX(iteration) FROM traineddata WHERE config_id = '" + config + "')";
            SQLQuery query = session.createSQLQuery(string);
            result = (Double) query.uniqueResult();
        } catch (NullPointerException e) {
            e.printStackTrace();
        }
        session.close();
        return result;
    }

    /**
     *
     * @param input
     * @param target
     * @param iteration
     */
    public void BackPropagation(double[] input, double[] target, int iteration) {

        double sum;

        FeedForward(input);
        /*
        find the delta for the output layer
        Dk <--- Ok * (1-Ok)(Tk - Ok)
         */
        for (int i = 0; i < this.layerSize[this.numLayer - 1]; i++) {
//            System.out.println("\nOutput["+(this.numLayer - 1)+"]["+i +"]"+this.out[this.numLayer - 1][i] + "  layerSize of output: "+this.layerSize[this.numLayer - 1] + " target: "+target[i] );

            this.delta[this.numLayer - 1][i] = this.out[this.numLayer - 1][i] * (1 - this.out[this.numLayer - 1][i]) * (target[i] - this.out[this.numLayer - 1][i]);
//            System.out.println("output deltavalue "+this.delta[this.numLayer - 1][i]);
        }
        /*
        find the delta for the hidden layer
        Hk <--- Oh * (1-Oh) summation Wkh * Dk       
         */

        for (int i = numLayer - 2; i > 0; i--) {
            for (int j = 0; j < layerSize[i]; j++) {
                sum = 0.0;
                for (int k = 0; k < layerSize[i + 1]; k++) {
                    sum += delta[i + 1][k] * weight[i + 1][k][j];
                }
                delta[i][j] = out[i][j] * (1 - out[i][j]) * sum;
            }
        }

//        apply momentum (does nothing if momentum = 0
        for (int i = 1; i < numLayer; i++) {
            for (int j = 0; j < layerSize[i]; j++) {
                for (int k = 0; k < layerSize[i - 1]; k++) {
                    weight[i][j][k] += momentum * prevWeight[i][j][k];
                }
                weight[i][j][layerSize[i - 1]] += momentum * prevWeight[i][j][layerSize[i - 1]];
            }
        }

//        adjust the weights by finding the correction to the weight 
//        
        System.out.println("Updated Weight:");
        for (int i = 1; i < numLayer; i++) {
            for (int j = 0; j < layerSize[i]; j++) {
                for (int k = 0; k < layerSize[i - 1]; k++) {
                    prevWeight[i][j][k] = learingRate * delta[i][j] * out[i - 1][k];
                    weight[i][j][k] += prevWeight[i][j][k];
                    objData.setLayer(i);
                    objData.setLayernode(j);
                    objData.setInputnode(k);
                    objData.setWeight(weight[i][j][k]);
                    objData.setIteration(iteration);
                    objData.setOutputvalue(0.0);
                    objData.setObjconfig(objConfig);
                    objTrain.insert(objData);
                    System.out.println("[" + i + "][" + j + "][" + k + "][" + this.weight[i][j][k] + "]");

                }
                prevWeight[i][j][layerSize[i - 1]] = learingRate * delta[i][j];
                weight[i][j][layerSize[i - 1]] += prevWeight[i][j][layerSize[i - 1]];

            }
        }

    }

    /**
     *
     * @param input
     */
    public void FeedForward(double[] input) {

        double sum;

//        System.out.println("layersize: "+this.layerSize[0]);
//assign content to input layer
        for (int i = 0; i < this.layerSize[0]; i++) {
            out[0][i] = input[i];

//            System.out.println("Input values[" +i+"]"+ out[0][i]);
        }

        // assign output(activation) value
        // to each neuron usng sigmoid function
        //for each layer
        System.out.println("Ouput value of each neuron :");
        for (int i = 1; i < this.numLayer; i++) {
            for (int j = 0; j < this.layerSize[i]; j++) {
                sum = 0.0;
                for (int k = 0; k < layerSize[i - 1]; k++) {
                    //apply weight to inputs and add to sum

                    sum += this.out[i - 1][k] * weight[i][j][k];
                }

                //apply bais 
                sum += weight[i][j][layerSize[i - 1]];

                //apply sigmoid function
                this.out[i][j] = activationFunction.calculateActivation(sum);

                objData.setLayer(i);
                objData.setInputnode(j);
                objData.setIteration(0);
                objData.setOutputvalue(out[i][j]);
                objTrain.insert(objData);

                System.out.println("Output[" + i + "][" + j + "]" + this.out[i][j]);
            }
        }

    }

    /**
     *
     * @param target
     * @return mean square error
     */
    public double MeanSquareError(double[] target) {

        double mse = 0;

        for (int i = 0; i < layerSize[numLayer - 1]; i++) {
            mse += (target[i] - out[numLayer - 1][i]) * (target[i] - out[numLayer - 1][i]);
        }

        return mse / 2;
    }

    /**
     *
     * @param i
     * @return returns i'th output of the net
     */
    public double Out(int i) {
        return out[numLayer - 1][i];
    }

}
