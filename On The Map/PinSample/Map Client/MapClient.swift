//
//  MapClient.swift
//  PinSample
//
//  Created by Ebraham Alskaf on 07/06/2024.
//  Copyright © 2024 Udacity. All rights reserved.
//

import Foundation
import UIKit

class MapClient {
    
    
    // Function to fetch Udacity user ID
    class func fetchUdacityUserId(completionHandler: @escaping (String?, Error?) -> Void) {
        let urlString = "http://quotes.rest/qod.json?category=inspire"
        guard let url = URL(string: urlString) else {
            completionHandler(nil, NSError(domain: "Invalid URL", code: -1, userInfo: nil))
            return
        }
        let request = URLRequest(url: url)
        let session = URLSession.shared
        let task = session.dataTask(with: request) { data, response, error in
            if let error = error {
                DispatchQueue.main.async {
                    completionHandler(nil, error)
                }
                return
            }
            guard let data = data else {
                DispatchQueue.main.async {
                    completionHandler(nil, NSError(domain: "No data", code: -1, userInfo: nil))
                }
                return
            }
            let quote = String(data: data, encoding: .utf8)
            DispatchQueue.main.async {
                completionHandler(quote, nil)
            }
        }
        task.resume()
    }
    
    
    class func requestStudentLocation<ResponseType: Decodable>(responseType: ResponseType.Type, completionHandler: @escaping (ResponseType?, Error?) -> Void) ->
    URLSessionTask {
        var request = URLRequest(url: URL(string: "https://onthemap-api.udacity.com/v1/StudentLocation?order=-updatedAt")!)
        let session = URLSession.shared
        let task = session.dataTask(with: request) {data, response, error in
            guard let data = data else {
                DispatchQueue.main.async {
                    completionHandler(nil, error)
                }
                return
            }
            let decoder = JSONDecoder()
            do {
                let responseObject = try decoder.decode(ResponseType.self, from: data)
                DispatchQueue.main.async {
                    completionHandler(responseObject, nil)
                }
            } catch {
                DispatchQueue.main.async {
                    completionHandler(nil, error)
                }
                
            }
        }
        
        task.resume()
        return task
    }
    
    
    class func postStudentLocation<ResponseType: Decodable, RequestBody: Encodable>(responseType: ResponseType.Type, body: RequestBody, completionHandler: @escaping(ResponseType?, Error?) -> Void) ->
    URLSessionTask {
        var request = URLRequest(url: URL(string: "https://onthemap-api.udacity.com/v1/StudentLocation")!)
        let session = URLSession.shared
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try! JSONEncoder().encode(body)
        let task = session.dataTask(with: request) {data, response, error in
            guard let data = data else {
                DispatchQueue.main.async {
                    completionHandler(nil, error)
                }
                return
            }
            let decoder = JSONDecoder()
            do {
                let responseObject = try decoder.decode(ResponseType.self, from: data)
                DispatchQueue.main.async {
                    completionHandler(responseObject, nil)
                }
            } catch {
                DispatchQueue.main.async {
                    completionHandler(nil, error)
                }
            }
        }
        task.resume()
        return task
        
    }
    
    
    class func updateStudentLocation<ResponseType: Decodable, RequestBody: Encodable>(responseType: ResponseType.Type, body: RequestBody, completionHandler: @escaping(ResponseType?, Error?) -> Void) ->
    URLSessionTask {
        var request = URLRequest(url: URL(string: "https://onthemap-api.udacity.com/v1/StudentLocation/8ZExGR5uX8")!)
        let session = URLSession.shared
        request.httpMethod = "PUT"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try! JSONEncoder().encode(body)
        let task = session.dataTask(with: request) {data, response, error in
            guard let data = data else {
                DispatchQueue.main.async {
                    completionHandler(nil, error)
                }
                return
            }
            let decoder = JSONDecoder()
            do {
                let responseObject = try decoder.decode(ResponseType.self, from: data)
                DispatchQueue.main.async {
                    completionHandler(responseObject, nil)
                }
            } catch {
                DispatchQueue.main.async {
                    completionHandler(nil, error)
                }
            }
        }
        task.resume()
        return task
    }
    
    
    class func postUdacitySession<ResponseType: Decodable, RequestBody: Encodable>(responseType: ResponseType.Type, body: RequestBody, completionHandler: @escaping(ResponseType?, Error?) -> Void) ->
    URLSessionTask {
        var request = URLRequest(url: URL(string: "https://onthemap-api.udacity.com/v1/session")!)
        let session = URLSession.shared
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Accept")
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try! JSONEncoder().encode(body)
        let task = session.dataTask(with: request) {data, response, error in
            guard let data = data else {
                DispatchQueue.main.async {
                    completionHandler(nil, error)
                }
                return
            }
            let range = 5..<data.count
            let newData = data.subdata(in: range)
            let decoder = JSONDecoder()
            do {
                let responseObject = try decoder.decode(ResponseType.self, from: newData)
                DispatchQueue.main.async {
                    completionHandler(responseObject, nil)
                }
            } catch {
                DispatchQueue.main.async {
                    completionHandler(nil, error)
                }
            }
        }
        task.resume()
        return task
    }
    
    class func deleteUdacitySession<ResponseType: Decodable>(responseType: ResponseType.Type, completionHandler: @escaping(ResponseType?, Error?) -> Void) ->
    URLSessionTask {
        var request = URLRequest(url: URL(string: "https://onthemap-api.udacity.com/v1/session")!)
        request.httpMethod = "DELETE"
        var xsrfCookie: HTTPCookie? = nil
        let sharedCookieStorage = HTTPCookieStorage.shared
        for cookie in sharedCookieStorage.cookies! {
            if cookie.name == "XSRF-TOKEN" {xsrfCookie = cookie}
        }
        if let xsrfCookie = xsrfCookie {
            request.setValue(xsrfCookie.value, forHTTPHeaderField: "X-XSRF-TOKEN")
        }
        let session = URLSession.shared
        let task = session.dataTask(with: request) {data, response, error in
            guard let data = data else {
                DispatchQueue.main.async {
                    completionHandler(nil, error)
                }
                return
            }
            let range = 5..<data.count
            let newData = data.subdata(in: range)
            let decoder = JSONDecoder()
            do {
                let responseObject = try decoder.decode(ResponseType.self, from: newData)
                DispatchQueue.main.async {
                    completionHandler(responseObject, nil)
                }
            } catch {
                DispatchQueue.main.async {
                    completionHandler(nil, error)
                }
            }
        }
        task.resume()
        return task
    }
    
    class func getUdacityUserData<ResponseType: Decodable>(responseType: ResponseType.Type, completionHandler: @escaping(ResponseType?, Error?) -> Void) ->
    URLSessionTask {
        var request = URLRequest(url: URL(string: "https://onthemap-api.udacity.com/v1/users/3903878747")!)
        let session = URLSession.shared
        let task = session.dataTask(with: request) {data, response, error in
            guard let data = data else {
                DispatchQueue.main.async {
                    completionHandler(nil, error)
                }
                return
            }
            let range = 5..<data.count
            let newData = data.subdata(in: range)
            let decoder = JSONDecoder()
            do {
                let responseObject = try decoder.decode(ResponseType.self, from: newData)
                DispatchQueue.main.async {
                    completionHandler(responseObject, nil)
                }
            } catch {
                DispatchQueue.main.async {
                    completionHandler(nil, error)
                }
            }
        }
        task.resume()
        return task
    }
    
    // Function to perform login
    class func login<ResponseType: Decodable, RequestBody: Encodable>(responseType: ResponseType.Type, body: RequestBody, completionHandler: @escaping (ResponseType?, Error?) -> Void) -> URLSessionDataTask {
        let urlString = "https://onthemap-api.udacity.com/v1/session"
        guard let url = URL(string: urlString) else {
            completionHandler(nil, NSError(domain: "Invalid URL", code: -1, userInfo: nil))
            return URLSessionDataTask() // Return an empty URLSessionDataTask
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Accept")
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try! JSONEncoder().encode(body)
        let session = URLSession.shared
        let task = session.dataTask(with: request) { data, response, error in
            guard let data = data else {
                DispatchQueue.main.async {
                    completionHandler(nil, error)
                }
                return
            }
            let range = 5..<data.count
            let newData = data.subdata(in: range)
            let decoder = JSONDecoder()
            do {
                let responseObject = try decoder.decode(ResponseType.self, from: newData)
                DispatchQueue.main.async {
                    completionHandler(responseObject, nil)
                }
            } catch {
                DispatchQueue.main.async {
                    completionHandler(nil, error)
                }
            }
        }
        task.resume()
        return task
    }
}
