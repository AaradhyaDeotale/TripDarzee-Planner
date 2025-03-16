import { Button } from '@/components/ui/button';
import Logo from '../assets/logo.png';
import { Input } from '@/components/ui/input';
import { AI_PROMPT, SelectBudgetOptions, SelectTravelesList } from '@/constans/options';
import { chatSession } from '@/service/AIModal';
import { useEffect, useState } from 'react';
import { toast } from 'sonner';
import { FcGoogle } from 'react-icons/fc';
import { AiOutlineLoading3Quarters } from 'react-icons/ai';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { useGoogleLogin } from '@react-oauth/google';
import axios from 'axios';
import { AutoComplete, Form } from 'antd';
import { doc, setDoc, getDocs, collection } from 'firebase/firestore';
import { db } from '@/service/firebaseConfig';
import { useNavigate } from 'react-router-dom';
import Chatbot from './Chatbot.jsx';
import UserTripCardItem from '@/components/my-trips/UserTripCardItem';
import { SolveTSPUsingPSO } from '@/lib/PSO_TSP';
import { calculateActualBudget } from '@/lib/budgetConstraint';

const GMAPS_API_KEY = import.meta.env.VITE_GMAPS_KEY;
const HOTEL_COST_ESTIMATE = {
  low: 2000,
  medium: 5000,
  high: 10000
};

function CreateTrip() {
  const [openDialog, setOpenDialog] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [formData, setFormData] = useState({
    location: '',
    noOfDays: '',
    budget: '',
    traveler: '',
  });

  const [options, setOptions] = useState([]);
  const [filteredOptions, setFilteredOptions] = useState([]);
  const [searchText, setSearchText] = useState('');
  const [communityTrips, setCommunityTrips] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    fetchCountriesAndCities();
    fetchCommunityTrips();
  }, []);

  const fetchCountriesAndCities = async () => {
    try {
      const countriesResponse = await axios.get('https://restcountries.com/v3.1/all');
      const countryOptions = countriesResponse.data.map((country) => ({
        value: country.name.common,
        label: country.name.common,
      }));

      const citiesResponse = await axios.get('https://countriesnow.space/api/v0.1/countries/population/cities');
      const cityOptions = citiesResponse.data.data.map((city) => ({
        value: city.city,
        label: `${city.city}, ${city.country}`,
      }));

      const allOptions = [...countryOptions, ...cityOptions];
      setOptions(allOptions);
      setFilteredOptions(allOptions);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  const fetchCommunityTrips = async () => {
    try {
      const querySnapshot = await getDocs(collection(db, 'AITrips'));
      const trips = querySnapshot.docs.map(doc => ({ id: doc.id, ...doc.data() }));
      setCommunityTrips(trips);
    } catch (error) {
      console.error('Error fetching community trips:', error);
    }
  };

  const handleSearch = (text) => {
    setSearchText(text);
    setFilteredOptions(
      text ? options.filter(option => 
        option.label.toLowerCase().includes(text.toLowerCase())
      ) : options
    );
  };
  
  const handleSelect = (value) => {
    setFormData({ ...formData, location: value });
    setSearchText(value);
  };

  const handleInputChange = (name, value) => {
    setFormData({ ...formData, [name]: value });
  };

  const login = useGoogleLogin({
    onSuccess: (codeResp) => GetUserProfile(codeResp),
    onError: (error) => console.log(error),
  });

  const onGenerateTrip = async () => {
    const user = JSON.parse(localStorage.getItem('user'));
    if (!user) return setOpenDialog(true);
    
    if (!formData.location || !formData.noOfDays || !formData.budget || !formData.traveler) {
      return toast.error('Please fill all details');
    }

    setIsLoading(true);
    toast.info('Creating your optimized trip...');

    try {
      const FINAL_PROMPT = AI_PROMPT
        .replace('{location}', formData.location)
        .replace('{totalDays}', formData.noOfDays)
        .replace('{traveler}', formData.traveler)
        .replace('{budget}', formData.budget);

      const result = await chatSession.sendMessage(FINAL_PROMPT);
      const tripDataText = result?.response?.text();

      // Parse and clean AI response
      const cleanedData = tripDataText
        .replace(/```json/g, '')
        .replace(/```/g, '')
        .replace(/,\s*}/g, '}')
        .replace(/,\s*\]/g, ']');
      const parsedData = JSON.parse(cleanedData);

      // Get coordinates for all locations
      const activities = parsedData.itinerary.flatMap(day => day.plan);
      const coordinates = await Promise.all(
        activities.map(async activity => {
          const res = await axios.get(
            `https://maps.googleapis.com/maps/api/geocode/json?address=${encodeURIComponent(activity.placeName)}&key=${GMAPS_API_KEY}`
          );
          const loc = res.data.results[0].geometry.location;
          return [loc.lat, loc.lng];
        })
      );

      // Run PSO optimization
      const pso = new SolveTSPUsingPSO({
        nodes: coordinates,
        populationSize: 20,
        iterations: 100
      });
      const { distance: totalDistance, route: optimizedRoute } = await pso.run();

      // Calculate budget constraints
      const hotelCost = HOTEL_COST_ESTIMATE[formData.budget];
      const actualBudget = calculateActualBudget(
        hotelCost,
        totalDistance,
        formData.noOfDays,
        parseInt(formData.traveler),
        formData.budget
      );

      // Enhance trip data with optimization results
      parsedData.optimization = {
        routeOrder: optimizedRoute,
        totalDistance: totalDistance.toFixed(2),
        estimatedCost: actualBudget.toFixed(2),
        coordinates
      };

      // Save enhanced trip
      await saveAiTrip(JSON.stringify(parsedData));

    } catch (error) {
      console.error('Trip creation failed:', error);
      toast.error(error.response?.data?.error || 'Failed to create trip');
    } finally {
      setIsLoading(false);
    }
  };

  const saveAiTrip = async (tripData) => {
    const user = JSON.parse(localStorage.getItem('user'));
    const docId = Date.now().toString();

    try {
      const parsedData = JSON.parse(tripData);
      await setDoc(doc(db, 'AITrips', docId), {
        userSelection: formData,
        tripData: parsedData,
        userEmail: user?.email,
        id: docId
      });
      navigate(`/view-trip/${docId}`);
    } catch (error) {
      console.error('Error saving trip:', error);
      toast.error('Failed to save trip data');
    }
  };

  const GetUserProfile = (tokenInfo) => {
    axios.get(`https://www.googleapis.com/oauth2/v1/userinfo?access_token=${tokenInfo.access_token}`, {
      headers: {
        Authorization: `Bearer ${tokenInfo.access_token}`,
        Accept: 'application/json'
      }
    }).then((resp) => {
      localStorage.setItem('user', JSON.stringify(resp.data));
      setOpenDialog(false);
      window.location.reload();
      onGenerateTrip();
    }).catch(error => {
      console.error('Google auth failed:', error);
      toast.error('Failed to authenticate with Google');
    });
  };

  return (
    <div className='sm:px-10 md:px-32 lg:px-96 xl:px-96 px-5 mt-10'>
      <h2 className='font-bold text-3xl'>Tell us your travel preferences 🏕️🌴</h2>
      <p className='mt-3 text-gray-500 text-xl'>Just provide some basic information, and our trip planner will generate a customized itinerary based on your preferences.</p>

      <div className="mt-20 flex flex-col gap-10">
        <div>
          <h2 className='text-xl my-3 font-medium'>What is your destination of choice?</h2>
          <div className="relative h-10 w-full">
            <Form labelCol={{ span: 12 }} wrapperCol={{ span: 24 }}>
              <Form.Item>
                <div style={{ display: 'flex' }}>
                  <AutoComplete
                    style={{ width: '100%' }}
                    options={filteredOptions}
                    value={searchText}
                    onChange={handleSearch}
                    onSelect={handleSelect}
                    placeholder="Search for a country or city"
                  />
                </div>
              </Form.Item>
            </Form>
          </div>
        </div>
        <div>
          <h2 className='text-xl my-3 font-medium'>How many days are you planning your trip?</h2>
          <div className="relative h-10 w-full">
            <Input
              placeholder={'Ex. 3'}
              type={'number'}
              onChange={(e) => handleInputChange('noOfDays', e.target.value)}
            />
          </div>
        </div>
      </div>

      <div className="mt-20">
        <h2 className='font-bold text-3xl'>What is Your Budget?</h2>
        <div className="grid grid-cols-3 gap-5 mt-5">
          {SelectBudgetOptions.map((item, index) => (
            <div
              className={`p-4 cursor-pointer border rounded-lg hover:shadow-lg ${formData?.budget === item.title ? 'shadow-lg border-black' : ''}`}
              key={index}
              onClick={() => handleInputChange('budget', item.title)}
            >
              <h2 className="text-4xl">{item.icon}</h2>
              <h2 className="font-bold text-lg">{item.title}</h2>
              <p>{item.description}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="mt-20">
        <h2 className='font-bold text-3xl'>How many people are traveling?</h2>
        <div className="grid grid-cols-3 gap-5 mt-5">
          {SelectTravelesList.map((item, index) => (
            <div
              className={`p-4 cursor-pointer border rounded-lg hover:shadow-lg ${formData?.traveler === item.title ? 'shadow-lg border-black' : ''}`}
              key={index}
              onClick={() => handleInputChange('traveler', item.title)}
            >
              <h2 className="text-4xl">{item.icon}</h2>
              <h2 className="font-bold text-lg">{item.title}</h2>
              <p>{item.description}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="flex justify-center my-14">
        <Button
          onClick={onGenerateTrip}
          disabled={isLoading}
          className={`bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          {isLoading ? <AiOutlineLoading3Quarters className='animate-spin' /> : 'Create Trip'}
        </Button>
      </div>

      <Chatbot />

      <div className='mt-20'>
        <h2 className='font-bold text-3xl'>Community Trips</h2>
        <div className='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5 mt-5'>
          {communityTrips.length > 0 ? (
            communityTrips.map((trip) => (
              <UserTripCardItem key={trip.id} trip={trip} />
            ))
          ) : (
            <p className='text-gray-500'>No community trips available yet.</p>
          )}
        </div>
      </div>

      <Dialog open={openDialog} onOpenChange={setOpenDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="flex justify-center">
              <img src={Logo} alt="Logo" className="h-10" />
            </DialogTitle>
            <DialogDescription className="flex justify-center">
              <Button onClick={login} className="gap-2">
                <FcGoogle className="h-7 w-7" />
                Sign in with Google
              </Button>
            </DialogDescription>
          </DialogHeader>
        </DialogContent>
      </Dialog>
    </div>
  );
}

export default CreateTrip;