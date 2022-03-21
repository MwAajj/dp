package classifier;

import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.neighboursearch.PerformanceStats;

import java.io.Serializable;
import java.util.Enumeration;

public class EuclideanDistance implements DistanceFunction, Serializable {
    private Instances instances;

    @Override
    public double distance(Instance first, Instance second) {
        if (first.numAttributes() != second.numAttributes()) {
            throw new RuntimeException("CalculateDistance: Incompatible size of instances");
        }
        double sum = 0;
        for (int i = 0; i < first.numAttributes(); i++) {
            double value = first.value(i);
            double value1 = second.value(i);
            double x = value - value1;
            if (Double.isNaN(x))
                continue;
            sum += Math.pow(x, 2);
        }
        return Math.sqrt(sum);
    }


    @Override
    public void setInstances(Instances insts) {
        instances = insts;
    }

    @Override
    public Instances getInstances() {
        return instances;
    }

    @Override
    public void setAttributeIndices(String value) {

    }

    @Override
    public String getAttributeIndices() {
        return null;
    }

    @Override
    public void setInvertSelection(boolean value) {

    }

    @Override
    public boolean getInvertSelection() {
        return false;
    }


    @Override
    public double distance(Instance first, Instance second, PerformanceStats stats)  {
        return distance(first, second);
    }

    @Override
    public double distance(Instance first, Instance second, double cutOffValue) {
        return distance(first, second);
    }

    @Override
    public double distance(Instance first, Instance second, double cutOffValue, PerformanceStats stats) {
        return distance(first, second);
    }

    @Override
    public void postProcessDistances(double[] distances) {

    }

    @Override
    public void update(Instance ins) {
        instances.add(ins);
    }

    @Override
    public void clean() {

    }

    @Override
    public Enumeration<Option> listOptions() {
        return null;
    }

    @Override
    public void setOptions(String[] options)  {

    }

    @Override
    public String[] getOptions() {
        return new String[0];
    }
}
