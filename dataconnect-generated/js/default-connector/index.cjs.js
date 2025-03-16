const { getDataConnect, validateArgs } = require('firebase/data-connect');

const connectorConfig = {
  connector: 'default',
  service: 'TripDarzee-Next-Gen-Travel-Planner-main',
  location: 'us-central1'
};
exports.connectorConfig = connectorConfig;

