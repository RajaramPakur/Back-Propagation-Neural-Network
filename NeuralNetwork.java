package stocks.marketanalysis.Neuralnetwork.neuralnetwork;

/**
 *
 * @author RajaramPakur
 */
public interface NeuralNetwork {
    
   /**
    * 
    * @param input
    * @param target
    * @param iteration 
    */
    public void BackPropagation(double[] input, double[] target, int iteration);
    
    /**
     * 
     * @param input 
     */
    public void FeedForward(double[] input);
    
    /**
     * 
     * @param target
     * @return double value of mean square error
     */
    public double MeanSquareError(double[] target);
    
    /**
     *
     * @param i
     * @return double value output of each node
     */
    public double Out(int i);
}
